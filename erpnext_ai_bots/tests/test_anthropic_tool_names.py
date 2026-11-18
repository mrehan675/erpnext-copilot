"""Tests for Anthropic-compatible tool names."""
import sys
import unittest
from types import ModuleType
from unittest.mock import patch


class TestAnthropicToolNames(unittest.TestCase):

    def test_schemas_use_api_compatible_names_and_resolve_back(self):
        with patch.dict(sys.modules, {"frappe": ModuleType("frappe")}):
            from erpnext_ai_bots.tools.registry import TOOL_MAP, ToolRegistry

            class FakeTool:
                def __init__(self, name):
                    self.name = name

                def schema(self):
                    return {
                        "name": self.name,
                        "description": "test",
                        "input_schema": {"type": "object", "properties": {}},
                    }

            registry = ToolRegistry.__new__(ToolRegistry)
            registry.get_tool = lambda name: FakeTool(name)

            schemas = registry.get_all_schemas()

            self.assertEqual(len(schemas), len(TOOL_MAP))
            for schema in schemas:
                self.assertRegex(schema["name"], r"^[a-zA-Z0-9_-]{1,128}$")
                self.assertEqual(
                    registry.get_namespaced_name_by_anthropic_name(schema["name"]),
                    schema["name"].replace("_", ".", 1),
                )

    def test_resolver_preserves_internal_names_and_rejects_unknown_names(self):
        with patch.dict(sys.modules, {"frappe": ModuleType("frappe")}):
            from erpnext_ai_bots.tools.registry import TOOL_MAP, ToolRegistry

            registry = ToolRegistry.__new__(ToolRegistry)

            for name in TOOL_MAP:
                self.assertEqual(
                    registry.get_namespaced_name_by_anthropic_name(name.replace(".", "_")),
                    name,
                )

            with self.assertRaises(ValueError):
                registry.get_namespaced_name_by_anthropic_name("unknown_tool")


if __name__ == "__main__":
    unittest.main()
