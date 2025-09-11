import os
from authlib.integrations.requests_client import OAuth2Session

class OAuth2Agent:
    PLATFORM_CONFIGS = {
        'slack': {
            'client_id': lambda: os.getenv('SLACK_CLIENT_ID'),
            'client_secret': lambda: os.getenv('SLACK_CLIENT_SECRET'),
            'authorize_url': 'https://slack.com/oauth/v2/authorize',
            'token_url': 'https://slack.com/api/oauth.v2.access',
            'scopes': ['channels:read', 'chat:write'],
            'redirect_uri': lambda: os.getenv('SLACK_REDIRECT_URI'),
            'token_params': {'include_granted_scopes': 'true'},
        },
        'microsoft': {
            'client_id': lambda: os.getenv('MS_CLIENT_ID'),
            'client_secret': lambda: os.getenv('MS_CLIENT_SECRET'),
            'authorize_url': 'https://login.microsoftonline.com/common/oauth2/v2.0/authorize',
            'token_url': 'https://login.microsoftonline.com/common/oauth2/v2.0/token',
            'scopes': ['https://graph.microsoft.com/.default'],
            'redirect_uri': lambda: os.getenv('MS_REDIRECT_URI'),
            'token_endpoint_auth_method': 'client_secret_post',
        },
        'gmail': {
            'client_id': lambda: os.getenv('GMAIL_CLIENT_ID'),
            'client_secret': lambda: os.getenv('GMAIL_CLIENT_SECRET'),
            'authorize_url': 'https://accounts.google.com/o/oauth2/auth',
            'token_url': 'https://oauth2.googleapis.com/token',
            'scopes': ['https://www.googleapis.com/auth/gmail.readonly'],
            'redirect_uri': lambda: os.getenv('GMAIL_REDIRECT_URI'),
            'access_type': 'offline',
            'prompt': 'consent',
        },
        'notion': {
            'client_id': lambda: os.getenv('NOTION_CLIENT_ID'),
            'client_secret': lambda: os.getenv('NOTION_CLIENT_SECRET'),
            'authorize_url': 'https://api.notion.com/v1/oauth/authorize',
            'token_url': 'https://api.notion.com/v1/oauth/token',
            'scopes': ['read', 'write'],
            'redirect_uri': lambda: os.getenv('NOTION_REDIRECT_URI'),
        },
    }

    def __init__(self, platform_name: str):
        platform = platform_name.lower()
        if platform not in self.PLATFORM_CONFIGS:
            raise ValueError(f"Unsupported platform: {platform_name}")
        self.platform = platform
        self.config = self._load_platform_config(platform)
        self.session = None

    def _load_platform_config(self, platform: str) -> dict:
        # Call lambdas to load env vars at runtime, avoid loading them on import
        config = self.PLATFORM_CONFIGS[platform].copy()
        for key, val in config.items():
            if callable(val):
                config[key] = val()
        return config

    def create_oauth_session(self, code_verifier: str = None, token: dict = None):
        client_kwargs = {}
        if self.platform == 'microsoft':
            client_kwargs['token_endpoint_auth_method'] = self.config.get('token_endpoint_auth_method', 'client_secret_basic')

        self.session = OAuth2Session(
            client_id=self.config['client_id'],
            client_secret=self.config['client_secret'],
            scope=self.config['scopes'],
            redirect_uri=self.config['redirect_uri'],
            code_verifier=code_verifier,
            token=token,
            **client_kwargs
        )

    def generate_authorization_url(self):
        self.create_oauth_session()
        # Merge extra params like token_params, access_type, prompt if any
        extra_params = {}
        if 'token_params' in self.config:
            extra_params.update(self.config['token_params'])
        # Add gmail specific params if present
        for param in ['access_type', 'prompt']:
            if param in self.config:
                extra_params[param] = self.config[param]

        uri, state = self.session.create_authorization_url(
            self.config['authorize_url'],
            **extra_params
        )
        return uri, state, getattr(self.session, 'code_verifier', None)

    def fetch_token(self, authorization_response_url: str, code_verifier: str = None):
        self.create_oauth_session(code_verifier=code_verifier)
        token = self.session.fetch_token(
            self.config['token_url'],
            authorization_response=authorization_response_url,
            client_secret=self.config['client_secret']
        )
        return token

    def refresh_access_token(self, refresh_token: str):
        self.create_oauth_session(token={'refresh_token': refresh_token})
        new_token = self.session.refresh_token(
            self.config['token_url'],
            refresh_token=refresh_token,
            client_id=self.config['client_id'],
            client_secret=self.config['client_secret']
        )
        return new_token


