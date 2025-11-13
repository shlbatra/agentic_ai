# Email Assistant Workflow

## Setup

1. Make sure you have the environment variable set in your `.env` file:
   ```
   M3_EMAIL_SERVER_API_URL=http://127.0.0.1:5000
   ```

2. Install dependencies:
   ```bash
   uv add "uvicorn[standard]"
   ```

3. Start the FastAPI email server:
   ```bash
   uv run uvicorn email_server.email_service:app --host 127.0.0.1 --port 5000 --reload
   ```

4. The server will be available at `http://127.0.0.1:5000`

## Usage

With the server running, you can now use the email functions in `utils.py` such as:
- `utils.test_send_email()`
- `utils.test_list_emails()`
- `utils.test_get_email(email_id)`