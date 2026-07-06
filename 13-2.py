# Standalone function (no leading indentation, no 'self' parameter, and closed brace)
def get_cors_config():
    """Get CORS configuration."""
    return {
        "allow_origins": ["*"],
        "allow_credentials": False,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }

# Call and print to test
if __name__ == "__main__":
    print(get_cors_config())
