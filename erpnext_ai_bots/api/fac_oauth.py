import base64
import hashlib
import secrets
from urllib.parse import urlencode

import frappe
import requests

from frappe import _
from frappe.utils import now_datetime, add_to_date, get_datetime


# ============================================================
# CONFIGURATION
# ============================================================


# def get_fac_base_url():
#     return frappe.db.get_single_value(
#         "Assistant Core Settings",
#         "site_url"
#     ).rstrip("/")

#FAC_BASE_URL = frappe.utils.get_url()
FAC_BASE_URL = "https://bq-demo.vsspk.com"


FAC_REGISTER_URL = (
    f"{FAC_BASE_URL}/api/method/"
    "frappe_assistant_core.api.oauth_registration.register_client"
)

FAC_AUTHORIZE_URL = (
    f"{FAC_BASE_URL}/api/method/"
    "frappe.integrations.oauth2.authorize"
)

FAC_TOKEN_URL = (
    f"{FAC_BASE_URL}/api/method/"
    "frappe.integrations.oauth2.get_token"
)

FAC_REVOKE_URL = (
    f"{FAC_BASE_URL}/api/method/"
    "frappe.integrations.oauth2.revoke_token"
)

FAC_MCP_URL = (
    f"{FAC_BASE_URL}/api/method/"
    "frappe_assistant_core.api.fac_endpoint.handle_mcp"
)

FAC_REDIRECT_URI = (
    f"{FAC_BASE_URL}/api/method/"
    "erpnext_ai_bots.api.fac_oauth.fac_oauth_callback"
)


# OAuth state is intentionally temporary.
# Tokens are NOT stored in cache.
OAUTH_STATE_TTL = 600


# ============================================================
# HELPERS
# ============================================================

def _state_cache_key(state):
    return f"fac_oauth_state:{state}"


def _generate_code_verifier():
    return (
        base64.urlsafe_b64encode(
            secrets.token_bytes(64)
        )
        .decode("utf-8")
        .rstrip("=")
    )


def _generate_code_challenge(code_verifier):

    digest = hashlib.sha256(
        code_verifier.encode("utf-8")
    ).digest()

    return (
        base64.urlsafe_b64encode(digest)
        .decode("utf-8")
        .rstrip("=")
    )


def _get_connection_name(user):

    return frappe.db.get_value(
        "FAC OAuth Connection",
        {
            "user": user
        },
        "name"
    )


def _get_connection(user):

    name = _get_connection_name(user)

    if not name:
        return None

    return frappe.get_doc(
        "FAC OAuth Connection",
        name
    )


def _get_password(doc, fieldname):

    try:

        return doc.get_password(
            fieldname,
            raise_exception=False
        )

    except Exception:

        return None


# ============================================================
# CONNECTION STATUS
# ============================================================

@frappe.whitelist()
def get_fac_connection_status():

    user = frappe.session.user

    if user == "Guest":

        return {
            "connected": False
        }

    connection = _get_connection(user)

    if not connection:

        return {
            "connected": False
        }

    has_refresh_token = bool(
        _get_password(
            connection,
            "refresh_token"
        )
    )

    has_access_token = bool(
        _get_password(
            connection,
            "access_token"
        )
    )

    connected = bool(
        connection.connected
        and (
            has_access_token
            or has_refresh_token
        )
    )

    return {
        "connected": connected,
        "user": user
    }


# ============================================================
# START OAUTH
# ============================================================

@frappe.whitelist()
def start_fac_oauth():

    user = frappe.session.user

    if user == "Guest":

        frappe.throw(
            _("Please login before connecting FAC.")
        )

    # --------------------------------------------------------
    # PKCE
    # --------------------------------------------------------

    code_verifier = (
        _generate_code_verifier()
    )

    code_challenge = (
        _generate_code_challenge(
            code_verifier
        )
    )

    state = secrets.token_urlsafe(32)

    # --------------------------------------------------------
    # Dynamic Client Registration
    # --------------------------------------------------------

    registration_payload = {

        "client_name":
            "ERPNext AI Copilot",

        "redirect_uris": [
            FAC_REDIRECT_URI
        ],

        "grant_types": [
            "authorization_code",
            "refresh_token"
        ],

        "response_types": [
            "code"
        ],

        "token_endpoint_auth_method":
            "none"
    }

    try:

        response = requests.post(
            FAC_REGISTER_URL,
            json=registration_payload,
            headers={
                "Accept":
                    "application/json",
                "Content-Type":
                    "application/json"
            },
            timeout=30
        )

        response.raise_for_status()

        client_data = response.json()

    except requests.RequestException:

        frappe.log_error(
            frappe.get_traceback(),
            "FAC OAuth Registration Error"
        )

        frappe.throw(
            _(
                "Unable to register OAuth client "
                "with FAC."
            )
        )

    client_id = client_data.get(
        "client_id"
    )

    if not client_id:

        frappe.log_error(
            frappe.as_json(
                client_data
            ),
            "FAC OAuth Missing Client ID"
        )

        frappe.throw(
            _(
                "FAC OAuth registration did not "
                "return a client ID."
            )
        )

    # --------------------------------------------------------
    # Store temporary OAuth transaction
    #
    # Redis is correct here because this information is
    # short-lived and single-use.
    # --------------------------------------------------------

    frappe.cache().set_value(

        _state_cache_key(state),

        {
            "user": user,
            "client_id": client_id,
            "code_verifier": code_verifier
        },

        expires_in_sec=OAUTH_STATE_TTL
    )

    # --------------------------------------------------------
    # Authorization URL
    # --------------------------------------------------------

    params = {

        "client_id":
            client_id,

        "response_type":
            "code",

        "scope":
            "all",

        "redirect_uri":
            FAC_REDIRECT_URI,

        "state":
            state,

        "code_challenge":
            code_challenge,

        "code_challenge_method":
            "S256"
    }

    authorization_url = (
        FAC_AUTHORIZE_URL
        + "?"
        + urlencode(params)
    )

    return {
        "authorization_url":
            authorization_url
    }


# ============================================================
# OAUTH CALLBACK
# ============================================================

@frappe.whitelist(allow_guest=True)
def fac_oauth_callback(
    code=None,
    state=None,
    error=None
):

    if error:

        frappe.throw(
            _(
                "FAC authorization failed: {0}"
            ).format(error)
        )

    if not code or not state:

        frappe.throw(
            _(
                "Invalid FAC OAuth callback."
            )
        )

    # --------------------------------------------------------
    # Retrieve and immediately delete state.
    #
    # This prevents replay of the OAuth callback.
    # --------------------------------------------------------

    cache_key = (
        _state_cache_key(state)
    )

    oauth_data = (
        frappe.cache().get_value(
            cache_key
        )
    )

    if not oauth_data:

        frappe.throw(
            _(
                "FAC OAuth session expired "
                "or state is invalid."
            )
        )

    frappe.cache().delete_value(
        cache_key
    )

    user = oauth_data.get(
        "user"
    )

    client_id = oauth_data.get(
        "client_id"
    )

    code_verifier = oauth_data.get(
        "code_verifier"
    )

    if (
        not user
        or not client_id
        or not code_verifier
    ):

        frappe.throw(
            _(
                "FAC OAuth session data "
                "is incomplete."
            )
        )

    # --------------------------------------------------------
    # Exchange code for tokens
    # --------------------------------------------------------

    payload = {

        "grant_type":
            "authorization_code",

        "code":
            code,

        "client_id":
            client_id,

        "redirect_uri":
            FAC_REDIRECT_URI,

        "code_verifier":
            code_verifier
    }

    try:

        response = requests.post(
            FAC_TOKEN_URL,
            data=payload,
            headers={
                "Accept":
                    "application/json",

                "Content-Type":
                    "application/x-www-form-urlencoded"
            },
            timeout=30
        )

        response.raise_for_status()

        token_data = response.json()

    except requests.RequestException:

        frappe.log_error(
            frappe.get_traceback(),
            "FAC OAuth Token Exchange Error"
        )

        frappe.throw(
            _(
                "Unable to obtain FAC "
                "OAuth access token."
            )
        )

    access_token = (
        token_data.get(
            "access_token"
        )
    )

    refresh_token = (
        token_data.get(
            "refresh_token"
        )
    )

    expires_in = int(
        token_data.get(
            "expires_in",
            3600
        )
    )

    if not access_token:

        frappe.throw(
            _(
                "FAC OAuth server did not "
                "return an access token."
            )
        )

    # --------------------------------------------------------
    # Calculate expiry
    #
    # 60-second safety buffer.
    # --------------------------------------------------------

    expires_at = add_to_date(
        now_datetime(),
        seconds=max(
            expires_in - 60,
            60
        ),
        as_datetime=True
    )

    # --------------------------------------------------------
    # Create/update encrypted connection
    # --------------------------------------------------------

    connection = (
        _get_connection(user)
    )

    if not connection:

        connection = frappe.get_doc({

            "doctype":
                "FAC OAuth Connection",

            "user":
                user,

            "connected":
                1
        })

    connection.client_id = (
        client_id
    )

    connection.access_token = (
        access_token
    )

    if refresh_token:

        connection.refresh_token = (
            refresh_token
        )

    connection.expires_at = (
        expires_at
    )

    connection.connected = 1

    connection.last_refreshed_at = (
        now_datetime()
    )

    if connection.is_new():

        connection.insert(
            ignore_permissions=True
        )

    else:

        connection.save(
            ignore_permissions=True
        )

    frappe.db.commit()

    # --------------------------------------------------------
    # Redirect back to ERPNext
    #
    # Change this to your actual route.
    # --------------------------------------------------------

    frappe.local.response["type"] = (
        "redirect"
    )

    frappe.local.response["location"] = (
        "/app"
    )


# ============================================================
# REFRESH TOKEN
# ============================================================

def refresh_fac_access_token(user):

    connection = (
        _get_connection(user)
    )

    if not connection:

        frappe.throw(
            _(
                "FAC is not connected. "
                "Please connect FAC first."
            )
        )

    refresh_token = (
        _get_password(
            connection,
            "refresh_token"
        )
    )

    if not refresh_token:

        connection.connected = 0

        connection.save(
            ignore_permissions=True
        )

        frappe.db.commit()

        frappe.throw(
            _(
                "FAC authorization has expired. "
                "Please reconnect FAC."
            )
        )

    payload = {

        "grant_type":
            "refresh_token",

        "refresh_token":
            refresh_token,

        "client_id":
            connection.client_id
    }

    try:

        response = requests.post(
            FAC_TOKEN_URL,
            data=payload,
            headers={
                "Accept":
                    "application/json",

                "Content-Type":
                    "application/x-www-form-urlencoded"
            },
            timeout=30
        )

        response.raise_for_status()

        token_data = response.json()

    except requests.RequestException:

        frappe.log_error(
            frappe.get_traceback(),
            "FAC OAuth Token Refresh Error"
        )

        frappe.throw(
            _(
                "Unable to refresh FAC authorization."
            )
        )

    access_token = (
        token_data.get(
            "access_token"
        )
    )

    if not access_token:

        frappe.throw(
            _(
                "FAC did not return "
                "a new access token."
            )
        )

    expires_in = int(
        token_data.get(
            "expires_in",
            3600
        )
    )

    expires_at = add_to_date(
        now_datetime(),
        seconds=max(
            expires_in - 60,
            60
        ),
        as_datetime=True
    )

    connection.access_token = (
        access_token
    )

    # OAuth providers may rotate refresh tokens.

    new_refresh_token = (
        token_data.get(
            "refresh_token"
        )
    )

    if new_refresh_token:

        connection.refresh_token = (
            new_refresh_token
        )

    connection.expires_at = (
        expires_at
    )

    connection.last_refreshed_at = (
        now_datetime()
    )

    connection.connected = 1

    connection.save(
        ignore_permissions=True
    )

    frappe.db.commit()

    return access_token


# ============================================================
# GET VALID TOKEN
# ============================================================

def get_valid_fac_access_token(
    user
):

    connection = (
        _get_connection(user)
    )

    if (
        not connection
        or not connection.connected
    ):

        frappe.throw(
            _(
                "FAC is not connected. "
                "Please connect FAC first."
            )
        )

    access_token = (
        _get_password(
            connection,
            "access_token"
        )
    )

    # --------------------------------------------------------
    # Token still valid
    # --------------------------------------------------------

    if (
        access_token
        and connection.expires_at
    ):

        expires_at = get_datetime(
            connection.expires_at
        )

        if (
            now_datetime()
            < expires_at
        ):

            return access_token

    # --------------------------------------------------------
    # Token expired.
    # Refresh automatically.
    # --------------------------------------------------------

    return refresh_fac_access_token(
        user
    )


# ============================================================
# DISCONNECT
# ============================================================

@frappe.whitelist()
def disconnect_fac():

    user = frappe.session.user

    connection = (
        _get_connection(user)
    )

    if not connection:

        return {
            "status":
                "disconnected"
        }

    access_token = (
        _get_password(
            connection,
            "access_token"
        )
    )

    # --------------------------------------------------------
    # Revoke token on OAuth server
    # --------------------------------------------------------

    if access_token:

        try:

            requests.post(
                FAC_REVOKE_URL,
                data={
                    "token":
                        access_token
                },
                headers={
                    "Content-Type":
                        "application/x-www-form-urlencoded"
                },
                timeout=15
            )

        except requests.RequestException:

            # Do not prevent local disconnect
            # if remote revocation fails.

            frappe.log_error(
                frappe.get_traceback(),
                "FAC OAuth Revocation Error"
            )

    # --------------------------------------------------------
    # Delete encrypted credentials
    # --------------------------------------------------------

    frappe.delete_doc(
        "FAC OAuth Connection",
        connection.name,
        ignore_permissions=True,
        force=True
    )

    frappe.db.commit()

    return {
        "status":
            "disconnected"
    }