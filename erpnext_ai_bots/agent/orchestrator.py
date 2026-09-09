import frappe
import json
import time
from erpnext_ai_bots.tools.registry import ToolRegistry
from erpnext_ai_bots.agent.streaming import StreamBridge
from erpnext_ai_bots.agent.subagent import SubagentSpawner
from erpnext_ai_bots.agent.prompts import get_system_prompt
from erpnext_ai_bots.guards.permissions import PermissionGuard
from erpnext_ai_bots.tools.sanitizer import InputSanitizer
from erpnext_ai_bots.utils.token_counter import TokenTracker
from erpnext_ai_bots.utils.prompt_defense import check_prompt_injection
from erpnext_ai_bots.api.fac_oauth import get_valid_fac_access_token

class Orchestrator:
    """Single agent that handles all user requests.
    Owns all tools. Streams responses. Spawns subagents only when needed.
    Supports both Anthropic and OpenAI (ChatGPT OAuth) providers.
    """

    def __init__(self, user: str, session_id: str, company: str = None):
        self.user = user
        self.session_id = session_id
        self.company = company or frappe.defaults.get_user_default("company", user)
        self.settings = frappe.get_cached_doc("AI Bot Settings")
        self.provider = self.settings.provider or "Anthropic"

        # Initialize components
        self.tool_registry = ToolRegistry(user=self.user, company=self.company)
        self.permission_guard = PermissionGuard(user=self.user)
        self.sanitizer = InputSanitizer()
        self.token_tracker = TokenTracker(session_id=self.session_id)
        self.stream_bridge = StreamBridge(session_id=self.session_id, user=self.user)

        # Conversation state
        self.messages = self._load_messages()
        self.turn_tool_calls = 0

    def _load_messages(self) -> list:
        session = frappe.get_doc("AI Chat Session", self.session_id)
        if session.messages_json:
            return json.loads(session.messages_json)
        return []

    def _save_messages(self):
        frappe.db.set_value(
            "AI Chat Session",
            self.session_id,
            {
                "messages_json": json.dumps(self.messages, default=str),
                "message_count": len([m for m in self.messages if m["role"] == "user"
                                      and not isinstance(m.get("content"), list)]),
                "last_message_at": frappe.utils.now_datetime(),
                "total_input_tokens": self.token_tracker.total_input,
                "total_output_tokens": self.token_tracker.total_output,
                "total_cost_usd": self.token_tracker.total_cost,
                "total_tool_calls": self.turn_tool_calls,
            },
            update_modified=False,
        )
        frappe.db.commit()

    def handle_message(self, user_message: str, image_url: str = None):
        """Main entry point. Called from the API endpoint."""
        # 1. Prompt injection defense
        check_prompt_injection(user_message)

        # 2. If an image is attached, include the URL in the text so the AI
        #    knows to call core_analyze_image. We don't send the image inline
        #    because private files can't be downloaded by the external API.
        if image_url:
            user_message = f"[Image attached at: {image_url}] {user_message}\n\nUse the core_analyze_image tool with image_url=\"{image_url}\" to analyze this image."
        msg_content = user_message

        self.messages.append({
            "role": "user",
            "content": msg_content,
            "timestamp": frappe.utils.now_datetime().isoformat(),
        })

        # 3. Route to the right provider
        if self.provider == "OpenAI":
            self._openai_standard_loop()
            
        elif self.provider == "OpenAI (ChatGPT OAuth)":
            self._openai_loop()
            
        elif self.provider == "OpenAI (FAC MCP)":
            self._openai_fac_mcp_loop()
            
        else:
            self._anthropic_loop()

        # 4. Persist
        self._save_messages()


    #openAI standard Loop
    def _openai_standard_loop(self):
            """Core agent loop using standard OpenAI API or OAuth tokens.
            
            Fully compatible with gpt-4o-mini.
            """
            try:
                import openai
            except ImportError:
                self.stream_bridge.send_error(
                    "OpenAI SDK not installed. Run: pip install openai"
                )
                return

            # Fetch authorization credential (API Key or OAuth access token)
            api_key = self.settings.get_password("api_key") if self.settings.api_key else None
            if not api_key:
                self.stream_bridge.send_error(
                    "No API credential configured. Set it up in your settings."
                )
                return

            # Initialize standard client
            client = openai.OpenAI(api_key=api_key)
            max_iterations = self.settings.max_tool_calls_per_turn or 15
            model_name = self.settings.model_name or "gpt-4o-mini"

            # Fix the 400 bad request error by wrapping the schemas dynamically
            raw_schemas = self.tool_registry.get_openai_schemas() or []
            formatted_openai_tools = []
            for schema in raw_schemas:
                if isinstance(schema, dict) and "function" in schema:
                    formatted_openai_tools.append(schema)
                else:
                    formatted_openai_tools.append({
                        "type": "function",
                        "function": {
                            "name": schema.get("name"),
                            "description": schema.get("description", ""),
                            "parameters": schema.get("parameters", {"type": "object", "properties": {}})
                        }
                    })

            for _ in range(max_iterations):
                # Builds standard user/assistant/tool history chain
                api_messages = self._prepare_messages_for_openai_api()
                
                system_prompt = get_system_prompt(self.user, self.company)
                messages_payload = [{"role": "system", "content": system_prompt}] + api_messages

                try:
                    stream = client.chat.completions.create(
                        model=model_name,
                        messages=messages_payload,
                        tools=formatted_openai_tools if formatted_openai_tools else None,
                        stream=True,
                        stream_options={"include_usage": True}
                    )

                    full_text = ""
                    tool_calls_map = {}
                    final_usage = {"input_tokens": 0, "output_tokens": 0}
                    finish_reason = None

                    for chunk in stream:
                        if not chunk.choices:
                            if hasattr(chunk, "usage") and chunk.usage:
                                final_usage = {
                                    "input_tokens": chunk.usage.prompt_tokens,
                                    "output_tokens": chunk.usage.completion_tokens
                                }
                            continue

                        choice = chunk.choices[0]
                        delta = choice.delta
                        
                        if choice.finish_reason:
                            finish_reason = choice.finish_reason

                        if getattr(delta, "content", None):
                            text_chunk = delta.content
                            full_text += text_chunk
                            self.stream_bridge._publish(
                                "ai_chunk",
                                {"session_id": self.session_id, "text": text_chunk},
                            )

                        if getattr(delta, "tool_calls", None):
                            for tc_delta in delta.tool_calls:
                                idx = tc_delta.index
                                if idx not in tool_calls_map:
                                    tool_calls_map[idx] = {
                                        "id": tc_delta.id,
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""}
                                    }
                                if tc_delta.function.name:
                                    tool_calls_map[idx]["function"]["name"] = tc_delta.function.name
                                if tc_delta.function.arguments:
                                    tool_calls_map[idx]["function"]["arguments"] += tc_delta.function.arguments

                    tool_calls_list = list(tool_calls_map.values())

                    self.token_tracker.record(
                        input_tokens=final_usage["input_tokens"],
                        output_tokens=final_usage["output_tokens"],
                        model=model_name,
                    )

                    assistant_message = {
                        "role": "assistant",
                        "content": full_text if full_text else None,
                        "timestamp": frappe.utils.now_datetime().isoformat(),
                        "usage": final_usage,
                    }
                    if tool_calls_list:
                        assistant_message["tool_calls"] = tool_calls_list

                    self.messages.append(assistant_message)

                    if finish_reason != "tool_calls" and not tool_calls_list:
                        self.stream_bridge.send_done()
                        break

                    # Process calls and return matching array items containing role: "tool"
                    tool_results = self._process_standard_openai_tool_calls(tool_calls_list)
                    self.messages.extend(tool_results)

                except Exception as e:
                    frappe.log_error(title="OpenAI Standard Loop Error", message=frappe.get_traceback())
                    self.stream_bridge.send_error(f"Execution Error: {str(e)}")
                    return
            else:
                self.stream_bridge.send_error(
                    "Maximum tool call limit reached. Please simplify your request."
                )

    def _prepare_messages_for_openai_api(self) -> list:
            """Build the conversation history list for the standard OpenAI Chat API
            from the stored conversation history (last 20 turns).

            Maintains the exact structured relationship required by OpenAI:
            - user: text
            - assistant: text and/or tool_calls
            - tool: tool execution outputs matching the tool_call_id
            """
            api_messages = []
            
            # Grab last 20 elements to respect context constraints
            for msg in self.messages[-6:]:
                role = msg["role"]
                
                # 1. Handle Tool Response Messages
                if role == "tool":
                    api_messages.append({
                        "role": "tool",
                        "tool_call_id": msg.get("tool_call_id"),
                        "name": msg.get("name"),
                        "content": msg.get("content", "")
                    })
                    continue

                # 2. Handle Assistant Messages (which might contain tool calls)
                if role == "assistant":
                    assistant_payload = {
                        "role": "assistant",
                        "content": msg.get("content") or None  # OpenAI allows None content if tool_calls exist
                    }
                    if "tool_calls" in msg:
                        assistant_payload["tool_calls"] = msg["tool_calls"]
                        
                    api_messages.append(assistant_payload)
                    continue

                # 3. Handle User Messages (with basic cleanup for legacy or mixed types)
                if role == "user":
                    content = msg.get("content", "")
                    
                    # Handle dictionary formats safely
                    if isinstance(content, dict):
                        content = content.get("text", str(content))
                        
                    # Extract plain text elements if it's passed as a list of blocks
                    elif isinstance(content, list):
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict):
                                if block.get("type") == "text":
                                    text_parts.append(block.get("text", ""))
                            else:
                                text_parts.append(str(block))
                        content = "\n".join(filter(None, text_parts))

                    if content:
                        api_messages.append({"role": "user", "content": content})

            return api_messages

    def _process_standard_openai_tool_calls(self, tool_calls_list: list) -> list:
        """Execute standard OpenAI function calls and return 'tool' role messages."""
        result_items = []

        for tc in tool_calls_list:
            # Clear accumulated messages so permission popups don't leak to client
            if hasattr(frappe.local, "message_log"):
                frappe.local.message_log = []

            self.turn_tool_calls += 1
            tool_call_id = tc["id"]
            function_name = tc["function"]["name"]
            
            # Convert OpenAI-safe name (underscores) back to dotted name
            # e.g. core_get_list -> core.get_list
            tool_name = function_name.replace("_", ".", 1)
            start_time = time.time()

            try:
                # Parse arguments — the model returns a JSON string
                tool_input = json.loads(tc["function"].get("arguments", "{}") or "{}")
            except (json.JSONDecodeError, ValueError):
                tool_input = {}

            self.stream_bridge.send_tool_start(tool_name, tool_input)

            try:
                # Permission check (same guard as Anthropic/Codex paths)
                self.permission_guard.check(tool_name, tool_input)
                
                # Input sanitization
                sanitized_input, blocked_fields = self.sanitizer.sanitize(tool_name, tool_input)

                # Subagent spawn or normal tool execution
                if tool_name == "meta.spawn_subagent":
                    result = self._handle_subagent(sanitized_input)
                else:
                    tool_fn = self.tool_registry.get_tool(tool_name)
                    result = tool_fn.execute(**sanitized_input)

                exec_time = int((time.time() - start_time) * 1000)

                # Audit Log
                self._audit_log(
                    tool_name=tool_name, 
                    tool_input=sanitized_input, 
                    tool_output=result,
                    status="Success", 
                    exec_time=exec_time, 
                    blocked_fields=blocked_fields
                )
                
                self.stream_bridge.send_tool_result(tool_name, result)

                # Append result in the exact format required by standard OpenAI API
                result_items.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "content": json.dumps(result, default=str),
                })

            except frappe.PermissionError as e:
                exec_time = int((time.time() - start_time) * 1000)
                self._audit_log(
                    tool_name=tool_name,
                    tool_input=tool_input,
                    tool_output={"error": str(e)},
                    status="PermissionDenied",
                    exec_time=exec_time,
                )
                result_items.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "content": json.dumps({"error": f"Permission denied: {e}"}, default=str),
                })

            except Exception as e:
                exec_time = int((time.time() - start_time) * 1000)
                self._audit_log(
                    tool_name=tool_name,
                    tool_input=tool_input,
                    tool_output={"error": str(e)},
                    status="Error",
                    exec_time=exec_time,
                )
                result_items.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "content": json.dumps({"error": str(e)}, default=str),
                })

        return result_items



    # ── OpenAI (ChatGPT OAuth) path ──────────────────────────────────

    def _openai_loop(self):
        """Agent loop using the ChatGPT Codex Responses API.

        Mirrors the Anthropic loop:
          1. Send messages + tool schemas to CodexClient.
          2. Stream text deltas to the frontend via stream_bridge.
          3. If the response contains function calls, execute each one with
             the same permission guard, sanitizer, and audit log used by the
             Anthropic path.
          4. Append tool results to the input and loop.
          5. Repeat until the model stops calling tools or max iterations hit.
        """
        from erpnext_ai_bots.licensing.openai_codex import CodexClient

        client = CodexClient(user=self.user)
        system_prompt = get_system_prompt(self.user, self.company)

        # Resolve model: only accept codex / gpt-5 slugs from settings.
        configured_model = self.settings.model_name or ""
        model = (
            configured_model
            if ("codex" in configured_model or configured_model.startswith("gpt-5"))
            else None
        )
        # model=None lets CodexClient fall back to DEFAULT_CODEX_MODEL.

        tool_schemas = self.tool_registry.get_openai_schemas()
        max_iterations = self.settings.max_tool_calls_per_turn or 15

        # The Codex Responses API keeps a flat input list across turns.
        # We seed it from conversation history on the first call, then
        # append tool-call items and tool-result items each iteration.
        api_input = self._build_openai_messages()

        try:
            for _ in range(max_iterations):
                result = client.send_streaming(
                    messages=api_input,
                    model=model,
                    instructions=system_prompt,
                    tools=tool_schemas,
                    on_delta=lambda delta: self.stream_bridge._publish(
                        "ai_chunk",
                        {"session_id": self.session_id, "text": delta},
                    ),
                )

                # Track token usage for this turn
                usage = result.get("usage", {})
                self.token_tracker.record(
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    model=model or "gpt-5.1-codex-mini",
                )

                function_calls = result.get("function_calls", [])
                output_items = result.get("output_items", [])
                response_text = result.get("text", "")

                # No function calls — final assistant turn
                if not function_calls:
                    self.messages.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": response_text}],
                        "timestamp": frappe.utils.now_datetime().isoformat(),
                        "usage": usage,
                    })
                    self.stream_bridge.send_done()
                    return

                # There are function calls — execute them and loop.
                # Persist a record of this assistant turn (tool-use turn).
                self.messages.append({
                    "role": "assistant",
                    "content": self._serialize_openai_output_items(
                        output_items, response_text
                    ),
                    "timestamp": frappe.utils.now_datetime().isoformat(),
                    "usage": usage,
                })

                # Extend the flat api_input with the assistant's output items
                # (which include the function_call items the model emitted).
                api_input.extend(
                    self._openai_output_items_for_input(output_items)
                )

                # Execute each tool call and collect results
                tool_result_items = self._process_openai_tool_calls(function_calls)

                # Append tool results to api_input for the next request
                api_input.extend(tool_result_items)

                # Persist tool results in conversation history
                self.messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "function_call_output",
                            "call_id": item["call_id"],
                            "output": item["output"],
                        }
                        for item in tool_result_items
                    ],
                    "timestamp": frappe.utils.now_datetime().isoformat(),
                })

            # Exhausted iterations without a final text response
            self.stream_bridge.send_error(
                "Maximum tool call limit reached. Please simplify your request."
            )

        except Exception as e:
            raw_error = str(e)
            # Log the technical error for debugging
            frappe.log_error(title="AI Oracle Error", message=frappe.get_traceback())
            # Show a user-friendly message
            friendly_msg = "I ran into an issue processing your request. Please try again."
            if "permission" in raw_error.lower():
                friendly_msg = "You don't have permission for that action. Ask your admin for access."
            elif "not found" in raw_error.lower():
                friendly_msg = "I couldn't find what you're looking for. Could you double-check the name?"
            elif "timeout" in raw_error.lower():
                friendly_msg = "The request took too long. Please try again with a simpler question."
            self.stream_bridge.send_error(friendly_msg)
            self.messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": friendly_msg}],
                "timestamp": frappe.utils.now_datetime().isoformat(),
            })

    def _build_openai_messages(self) -> list:
        """Build the initial flat input list for the Codex Responses API
        from the stored conversation history (last 20 turns).

        Only plain user/assistant text messages are included. Tool call
        history (function_call and function_call_output items) from prior
        requests are EXCLUDED because the Codex Responses API treats each
        request independently — it doesn't remember previous function calls,
        so sending orphaned tool results causes "No tool call found" errors.
        """
        api_messages = []
        for msg in self.messages[-6:]:
            role = msg["role"]
            content = msg.get("content", "")

            # Handle legacy dict content (e.g. old vision messages)
            if isinstance(content, dict):
                content = content.get("text", str(content))

            if isinstance(content, list):
                # Skip turns that contain function_call_output items
                # (tool results from previous loop iterations).
                has_tool_content = any(
                    isinstance(b, dict) and b.get("type") in (
                        "function_call_output", "function_call"
                    )
                    for b in content
                )
                if has_tool_content:
                    continue

                # Extract plain text parts
                text_parts = [
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                content = "\n".join(filter(None, text_parts))

            if content:
                api_messages.append({"role": role, "content": content})

        return api_messages

    def _serialize_openai_output_items(
        self, output_items: list, response_text: str
    ) -> list:
        """Convert raw Codex output items into JSON-serialisable content blocks
        for storage in ``self.messages``.

        Text output is stored as ``{"type": "text", "text": "..."}`` and
        function-call items are stored verbatim (they are already plain dicts).
        """
        blocks = []
        if response_text:
            blocks.append({"type": "text", "text": response_text})
        for item in output_items:
            if isinstance(item, dict) and item.get("type") == "function_call":
                blocks.append(item)
        return blocks

    def _openai_output_items_for_input(self, output_items: list) -> list:
        """Return function_call items formatted for the Codex API input.

        The API requires these exact fields when echoing function calls back:
        type, id, call_id, name, arguments. Extra fields cause errors.
        """
        items = []
        for item in output_items:
            if isinstance(item, dict) and item.get("type") == "function_call":
                items.append({
                    "type": "function_call",
                    "id": item.get("id", ""),
                    "call_id": item.get("call_id", ""),
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "{}"),
                })
        return items

    def _process_openai_tool_calls(self, function_calls: list) -> list:
        """Execute OpenAI function calls with permission checks, sanitization,
        and audit logging — the same guards as the Anthropic path.

        Args:
            function_calls: List of dicts produced by ``_consume_stream``::

                [{"id": "fc_...", "call_id": "call_...",
                  "name": "tool_name", "arguments": "{...}"}]

        Returns:
            List of ``function_call_output`` dicts ready to be appended to
            the Codex API input array::

                [{"type": "function_call_output",
                  "call_id": "call_...",
                  "output": "{\"result\": ...}"}]
        """
        result_items = []

        for fc in function_calls:
            # Clear any accumulated messages from the previous tool call so
            # permission popups and msgprint output don't leak to the client.
            if hasattr(frappe.local, "message_log"):
                frappe.local.message_log = []

            self.turn_tool_calls += 1
            openai_name = fc["name"]
            # Convert OpenAI-safe name (underscores) back to dotted name
            # e.g. core_get_list -> core.get_list
            tool_name = openai_name.replace("_", ".", 1)
            call_id = fc["call_id"]
            start_time = time.time()

            # Parse arguments — the model returns a JSON string
            try:
                tool_input = json.loads(fc.get("arguments", "{}") or "{}")
            except (json.JSONDecodeError, ValueError):
                tool_input = {}

            self.stream_bridge.send_tool_start(tool_name, tool_input)

            try:
                # Permission check (same guard as Anthropic path)
                self.permission_guard.check(tool_name, tool_input)

                # Input sanitization
                sanitized_input, blocked_fields = self.sanitizer.sanitize(
                    tool_name, tool_input
                )

                # Subagent spawn
                if tool_name == "meta.spawn_subagent":
                    result = self._handle_subagent(sanitized_input)
                else:
                    tool_fn = self.tool_registry.get_tool(tool_name)
                    result = tool_fn.execute(**sanitized_input)

                exec_time = int((time.time() - start_time) * 1000)

                self._audit_log(
                    tool_name=tool_name,
                    tool_input=sanitized_input,
                    tool_output=result,
                    status="Success",
                    exec_time=exec_time,
                    blocked_fields=blocked_fields,
                )

                self.stream_bridge.send_tool_result(tool_name, result)

                result_items.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(result, default=str),
                })

            except frappe.PermissionError as e:
                exec_time = int((time.time() - start_time) * 1000)
                self._audit_log(
                    tool_name=tool_name,
                    tool_input=tool_input,
                    tool_output={"error": str(e)},
                    status="PermissionDenied",
                    exec_time=exec_time,
                )
                result_items.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(
                        {"error": f"Permission denied: {e}"}, default=str
                    ),
                })

            except Exception as e:
                exec_time = int((time.time() - start_time) * 1000)
                self._audit_log(
                    tool_name=tool_name,
                    tool_input=tool_input,
                    tool_output={"error": str(e)},
                    status="Error",
                    exec_time=exec_time,
                )
                result_items.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps({"error": str(e)}, default=str),
                })

        return result_items


    # ── FAC OpenAi path ──────────────────
    def _openai_fac_mcp_loop(self):
        """
        OpenAI Responses API using FAC as a remote MCP server.

        Authentication:
            OpenAI API Key -> OpenAI
            FAC OAuth Token -> FAC MCP as current ERPNext user

        Performance tracking:
            - Token retrieval time
            - History preparation time
            - Time to first text token
            - OpenAI + FAC MCP total request time
            - Complete method execution time
        """

        method_start = time.time()

        # =========================================================
        # 1. IMPORT REQUIRED MODULES
        # =========================================================

        try:
            from openai import OpenAI
            from erpnext_ai_bots.api.fac_oauth import get_valid_fac_access_token

        except ImportError:
            frappe.log_error(
                title="FAC MCP Import Error",
                message=frappe.get_traceback(),
            )

            self.stream_bridge.send_error(
                "Required OpenAI/FAC integration modules are not available."
            )
            return

        # =========================================================
        # 2. GET OPENAI API KEY
        # =========================================================

        api_key = (
            self.settings.get_password("api_key")
            if self.settings.api_key
            else None
        )

        if not api_key:
            self.stream_bridge.send_error(
                "OpenAI API Key is not configured."
            )
            return

        # =========================================================
        # 3. GET VALID FAC OAUTH ACCESS TOKEN
        # =========================================================

        token_start = time.time()

        try:
            fac_access_token = get_valid_fac_access_token(self.user)

        except Exception:
            frappe.log_error(
                title="FAC OAuth Token Error",
                message=frappe.get_traceback(),
            )

            self.stream_bridge.send_error(
                "Unable to authenticate with FAC. Please reconnect FAC."
            )
            return

        token_time = time.time() - token_start

        if not fac_access_token:
            self.stream_bridge.send_error(
                "FAC is not connected. Please connect FAC first."
            )
            return

        # =========================================================
        # 4. INITIALIZE OPENAI CLIENT
        # =========================================================

        client = OpenAI(
            api_key=api_key,
            timeout=120.0,
            max_retries=1,
        )

        # =========================================================
        # 5. MODEL
        # =========================================================

        model_name = self.settings.model_name or "gpt-4.1"

        # =========================================================
        # 6. SYSTEM INSTRUCTIONS
        # Keep prompt short to reduce unnecessary input tokens.
        # =========================================================

        system_prompt = (
            "You are an AI assistant integrated with ERPNext through "
            "Frappe Assistant Core (FAC) MCP. "
            "Use FAC MCP tools whenever current ERPNext data is required. "
            "Never invent ERPNext records or business data. "
            "Respect the authenticated user's ERPNext permissions. "
            "If data cannot be retrieved, clearly say so."
        )

        if self.company:
            system_prompt += (
                f" The current company is '{self.company}'. "
                "Use it as the default company when a company filter is required."
            )

        # =========================================================
        # 7. BUILD CONVERSATION HISTORY
        # =========================================================

        history_start = time.time()

        api_input = []

        # Keep only recent messages.
        # Your current code already uses 6.
        # This helps reduce TPM usage and request size.
        for msg in self.messages[-6:]:

            role = msg.get("role")
            content = msg.get("content", "")

            if role not in ("user", "assistant"):
                continue

            # -----------------------------------------
            # Dictionary content
            # -----------------------------------------

            if isinstance(content, dict):
                content = content.get("text", "")

            # -----------------------------------------
            # List/block content
            # -----------------------------------------

            elif isinstance(content, list):

                text_parts = []

                for block in content:

                    if isinstance(block, dict):

                        if block.get("type") == "text":

                            text = block.get("text", "")

                            if text:
                                text_parts.append(text)

                    elif isinstance(block, str):
                        text_parts.append(block)

                content = "\n".join(text_parts)

            # -----------------------------------------
            # Add valid conversation message
            # -----------------------------------------

            if content:

                # Prevent extremely large historical messages
                # from unnecessarily increasing request tokens.
                if len(content) > 12000:
                    content = content[-12000:]

                api_input.append({
                    "role": role,
                    "content": content,
                })

        history_time = time.time() - history_start

        if not api_input:
            self.stream_bridge.send_error(
                "No conversation message was available to process."
            )
            return

        # =========================================================
        # 8. FAC MCP ENDPOINT
        # =========================================================

        assistant_settings = frappe.get_cached_doc("Assistant Core Settings")

        fac_mcp_url = assistant_settings.mcp_endpoint_url
        # fac_mcp_url = (
        #     "https://erp-demo.almominpackaging.com"
        #     "/api/method/"
        #     "frappe_assistant_core.api.fac_endpoint.handle_mcp"
        # )

        # =========================================================
        # 9. START OPENAI + FAC MCP REQUEST
        # =========================================================

        request_start = time.time()

        full_text = ""

        first_token_time = None

        try:

            with client.responses.stream(

                model=model_name,

                input=api_input,

                instructions=system_prompt,

                tools=[
                    {
                        "type": "mcp",
                        "server_label": "fac_erpnext",
                        "server_url": fac_mcp_url,
                        "authorization": fac_access_token,
                        "require_approval": "never",
                    }
                ],

            ) as stream:

                # =================================================
                # 10. PROCESS STREAM EVENTS
                # =================================================

                for event in stream:

                    event_type = getattr(
                        event,
                        "type",
                        ""
                    )

                    # ---------------------------------------------
                    # Text streaming
                    # ---------------------------------------------

                    if event_type == "response.output_text.delta":

                        delta = getattr(
                            event,
                            "delta",
                            ""
                        ) or ""

                        if delta:

                            # Record time until user sees first text
                            if first_token_time is None:
                                first_token_time = (
                                    time.time() - request_start
                                )

                            full_text += delta

                            # Immediately push text to ERPNext widget
                            self.stream_bridge._publish(
                                "ai_chunk",
                                {
                                    "session_id": self.session_id,
                                    "text": delta,
                                },
                            )

                # =================================================
                # 11. GET FINAL RESPONSE
                # =================================================

                final_response = stream.get_final_response()

            request_time = time.time() - request_start

            # =====================================================
            # 12. FALLBACK IF STREAM DID NOT RETURN TEXT DELTAS
            # =====================================================

            if not full_text:

                full_text = (
                    getattr(
                        final_response,
                        "output_text",
                        "",
                    )
                    or ""
                )

                if full_text:

                    self.stream_bridge._publish(
                        "ai_chunk",
                        {
                            "session_id": self.session_id,
                            "text": full_text,
                        },
                    )

            # =====================================================
            # 13. EMPTY RESPONSE FALLBACK
            # =====================================================

            if not full_text:

                full_text = (
                    "The request was processed, but no "
                    "text response was returned."
                )

                self.stream_bridge._publish(
                    "ai_chunk",
                    {
                        "session_id": self.session_id,
                        "text": full_text,
                    },
                )

            # =====================================================
            # 14. SAVE ASSISTANT RESPONSE
            # =====================================================

            self.messages.append({
                "role": "assistant",
                "content": full_text,
                "timestamp": frappe.utils.now_datetime().isoformat(),
            })

            # =====================================================
            # 15. COMPLETE STREAM
            # =====================================================

            self.stream_bridge.send_done()

            # =====================================================
            # 16. PERFORMANCE LOG
            # =====================================================

            total_method_time = time.time() - method_start

            first_token_display = (
                f"{first_token_time:.2f}"
                if first_token_time is not None
                else "N/A"
            )

            frappe.log_error(
                title="FAC Performance Report",
                message=(
                    f"User: {self.user}\n"
                    f"Model: {model_name}\n"
                    f"History Messages: {len(api_input)}\n"
                    f"Token Retrieval: {token_time:.2f} seconds\n"
                    f"History Preparation: {history_time:.2f} seconds\n"
                    f"Time To First Text Token: "
                    f"{first_token_display} seconds\n"
                    f"OpenAI + FAC MCP Request: "
                    f"{request_time:.2f} seconds\n"
                    f"Total Method Time: "
                    f"{total_method_time:.2f} seconds"
                ),
            )

        # =========================================================
        # 17. ERROR HANDLING
        # =========================================================

        except Exception as e:

            failed_after = time.time() - request_start

            frappe.log_error(
                title="OpenAI FAC MCP Error",
                message=(
                    f"Request failed after: "
                    f"{failed_after:.2f} seconds\n\n"
                    f"{frappe.get_traceback()}"
                ),
            )

            raw_error = str(e).lower()

            # ---------------------------------------------
            # FAC / OAuth authentication
            # ---------------------------------------------

            if (
                "401" in raw_error
                or "unauthorized" in raw_error
                or "invalid_token" in raw_error
            ):

                friendly_msg = (
                    "Your FAC connection is no longer authorized. "
                    "Please reconnect FAC."
                )

            # ---------------------------------------------
            # Rate limit
            # IMPORTANT: Check before generic MCP error.
            # ---------------------------------------------

            elif (
                "429" in raw_error
                or "rate limit" in raw_error
                or "rate_limit" in raw_error
                or "tokens per min" in raw_error
            ):

                friendly_msg = (
                    "The AI service rate limit has been reached. "
                    "Please wait a few seconds and try again."
                )

            # ---------------------------------------------
            # Timeout
            # ---------------------------------------------

            elif (
                "timeout" in raw_error
                or "timed out" in raw_error
            ):

                friendly_msg = (
                    "The ERPNext request took too long. "
                    "Please try again."
                )

            # ---------------------------------------------
            # MCP communication
            # ---------------------------------------------

            elif "mcp" in raw_error:

                friendly_msg = (
                    "I could not communicate with the FAC MCP server. "
                    "Please try again or contact your administrator."
                )

            # ---------------------------------------------
            # Generic error
            # ---------------------------------------------

            else:

                friendly_msg = (
                    "I ran into an issue while communicating "
                    "with the ERPNext AI service."
                )

            # =====================================================
            # 18. SEND ERROR TO CHAT WIDGET
            # =====================================================

            self.stream_bridge.send_error(
                friendly_msg
            )

            # =====================================================
            # 19. SAVE ERROR MESSAGE
            # =====================================================

            self.messages.append({
                "role": "assistant",
                "content": friendly_msg,
                "timestamp": frappe.utils.now_datetime().isoformat(),
            })

    # ── Anthropic path ───────────────────────────────────────────────

    def _anthropic_loop(self):
        """Core loop using Anthropic API: call model, stream response, handle tool calls."""
        try:
            import anthropic
        except ImportError:
            self.stream_bridge.send_error(
                "Anthropic SDK not installed. Run: pip install anthropic"
            )
            return

        api_key = self.settings.get_password("api_key") if self.settings.api_key else None
        if not api_key:
            self.stream_bridge.send_error(
                "No API key configured. Set it in AI Bot Settings."
            )
            return

        client = anthropic.Anthropic(api_key=api_key)
        max_iterations = self.settings.max_tool_calls_per_turn or 15

        for _ in range(max_iterations):
            api_messages = self._prepare_messages_for_api()

            with client.messages.stream(
                model=self.settings.model_name or "claude-sonnet-4-20250514",
                max_tokens=self.settings.max_tokens_per_request or 4096,
                system=get_system_prompt(self.user, self.company),
                tools=self.tool_registry.get_all_schemas(),
                messages=api_messages,
            ) as stream:
                response = self.stream_bridge.process_stream(stream)

            # Track tokens
            self.token_tracker.record(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_creation_tokens=getattr(
                    response.usage, "cache_creation_input_tokens", 0
                ),
                cache_read_tokens=getattr(
                    response.usage, "cache_read_input_tokens", 0
                ),
                model=self.settings.model_name,
            )

            # Append assistant response
            self.messages.append({
                "role": "assistant",
                "content": self._serialize_content_blocks(response.content),
                "timestamp": frappe.utils.now_datetime().isoformat(),
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            })

            if response.stop_reason != "tool_use":
                self.stream_bridge.send_done()
                break

            # Process tool calls
            tool_results = self._process_tool_calls(response.content)
            self.messages.append({
                "role": "user",
                "content": tool_results,
                "timestamp": frappe.utils.now_datetime().isoformat(),
            })
        else:
            self.stream_bridge.send_error(
                "Maximum tool call limit reached. Please simplify your request."
            )

    def _prepare_messages_for_api(self) -> list:
        """Strip custom metadata keys before sending to Anthropic."""
        cleaned = []
        for msg in self.messages:
            cleaned.append({"role": msg["role"], "content": msg["content"]})

        # Context window management: cap at 100 messages
        if len(cleaned) > 100:
            cleaned = cleaned[:2] + cleaned[-98:]

        return cleaned

    def _serialize_content_blocks(self, content_blocks) -> list:
        """Convert Anthropic SDK content block objects to JSON-serializable dicts."""
        serialized = []
        for block in content_blocks:
            if block.type == "text":
                serialized.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                serialized.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        return serialized

    def _process_tool_calls(self, content_blocks) -> list:
        """Execute tool calls with permission checks, sanitization, and audit."""
        results = []

        for block in content_blocks:
            if block.type != "tool_use":
                continue

            # Clear accumulated messages so permission popups don't leak to client
            if hasattr(frappe.local, "message_log"):
                frappe.local.message_log = []

            self.turn_tool_calls += 1
            tool_name = block.name
            tool_input = block.input
            tool_id = block.id
            start_time = time.time()

            self.stream_bridge.send_tool_start(tool_name, tool_input)

            try:
                # Permission check
                self.permission_guard.check(tool_name, tool_input)

                # Input sanitization
                sanitized_input, blocked_fields = self.sanitizer.sanitize(
                    tool_name, tool_input
                )

                # Subagent spawn
                if tool_name == "meta.spawn_subagent":
                    result = self._handle_subagent(sanitized_input)
                else:
                    tool_fn = self.tool_registry.get_tool(tool_name)
                    result = tool_fn.execute(**sanitized_input)

                exec_time = int((time.time() - start_time) * 1000)

                self._audit_log(
                    tool_name=tool_name,
                    tool_input=sanitized_input,
                    tool_output=result,
                    status="Success",
                    exec_time=exec_time,
                    blocked_fields=blocked_fields,
                )

                self.stream_bridge.send_tool_result(tool_name, result)

                results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": json.dumps(result, default=str),
                })

            except frappe.PermissionError as e:
                exec_time = int((time.time() - start_time) * 1000)
                self._audit_log(
                    tool_name=tool_name,
                    tool_input=tool_input,
                    tool_output={"error": str(e)},
                    status="PermissionDenied",
                    exec_time=exec_time,
                )
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": json.dumps({"error": f"Permission denied: {e}"}),
                    "is_error": True,
                })

            except Exception as e:
                exec_time = int((time.time() - start_time) * 1000)
                self._audit_log(
                    tool_name=tool_name,
                    tool_input=tool_input,
                    tool_output={"error": str(e)},
                    status="Error",
                    exec_time=exec_time,
                )
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": json.dumps({"error": str(e)}),
                    "is_error": True,
                })

        return results

    def _handle_subagent(self, params: dict) -> dict:
        if not self.settings.subagent_enabled:
            return {"error": "Subagent spawning is disabled"}

        spawner = SubagentSpawner(
            user=self.user,
            company=self.company,
            parent_session_id=self.session_id,
            token_tracker=self.token_tracker,
            stream_bridge=self.stream_bridge,
            max_depth=self.settings.max_subagent_depth or 2,
        )
        return spawner.run(
            task_description=params["task"],
            tools_needed=params.get("tools", []),
            context=params.get("context", ""),
        )

    def _audit_log(self, tool_name, tool_input, tool_output,
                   status, exec_time, blocked_fields=None):
        """Write an immutable audit log entry."""
        action_type = "Read"
        if "create" in tool_name:
            action_type = "Create"
        elif "update" in tool_name:
            action_type = "Update"
        elif "submit" in tool_name:
            action_type = "Submit"

        output_str = json.dumps(tool_output, default=str)
        if len(output_str) > 10000:
            output_str = output_str[:10000] + "... [truncated]"

        frappe.get_doc({
            "doctype": "AI Audit Log",
            "session": self.session_id,
            "user": self.user,
            "company": self.company,
            "tool_name": tool_name,
            "tool_input_json": json.dumps(tool_input, default=str),
            "tool_output_json": output_str,
            "tool_result_status": status,
            "execution_time_ms": exec_time,
            "doctype_accessed": tool_input.get("doctype", ""),
            "document_name": tool_input.get("name", ""),
            "action_type": action_type,
            "fields_requested": json.dumps(tool_input.get("fields", [])),
            "fields_blocked": json.dumps(blocked_fields or []),
        }).insert(ignore_permissions=True)
        frappe.db.commit()


def run_orchestrator(user: str, session_id: str, message: str, company: str, image_url: str = None):
    """Entry point for background job execution."""
    frappe.set_user(user)
    # Suppress Frappe's popup messages — errors are shown in the AI chat instead.
    # Without this, permission errors from has_permission(throw=True) would show
    # as browser popups in addition to the chat error.
    frappe.flags.mute_messages = True
    if hasattr(frappe.local, "message_log"):
        frappe.local.message_log = []
    try:
        agent = Orchestrator(user=user, session_id=session_id, company=company)
        agent.handle_message(message, image_url=image_url)
    except Exception as e:
        # Clear any accumulated messages so they don't leak to the client
        if hasattr(frappe.local, "message_log"):
            frappe.local.message_log = []
        # Log the full technical error for developers
        frappe.log_error(title="AI Agent Error", message=frappe.get_traceback())

        # Determine a friendly message — never expose raw errors to users
        raw_error = str(e).lower()
        if "permission" in raw_error:
            friendly_msg = "You don't have permission for that action. Ask your admin for access."
        elif "not found" in raw_error:
            friendly_msg = "I couldn't find what you're looking for. Could you double-check the name?"
        elif "timeout" in raw_error or "timed out" in raw_error:
            friendly_msg = "The request took too long. Please try again with a simpler question."
        elif "rate limit" in raw_error or "rate_limit" in raw_error:
            friendly_msg = "The AI service is temporarily busy. Please wait a moment and try again."
        elif "api key" in raw_error or "authentication" in raw_error:
            friendly_msg = "There is a configuration issue with the AI service. Please contact your administrator."
        else:
            friendly_msg = "I ran into an issue processing your request. Please try again."

        frappe.publish_realtime(
            event="ai_error",
            message={"session_id": session_id, "error": friendly_msg},
            user=user,
            after_commit=False,
        )
