# Development and production credentials

The AI process has one active `ENVIRONMENT` per deployment. Backend
registrations may be marked `development` or `production`; authentication only
considers registrations matching the active environment. A production process
cannot authenticate a development key, and vice versa.

Use separate `.env` files or secret stores. Do not put both environments'
plaintext keys in the same deployment secret store.

Production example:

```dotenv
ENVIRONMENT=production
BACKEND_REGISTRY_JSON={"backend-a":{"environment":"production","service_key_sha256":"<sha256>","allowed_endpoint_urls":["https://prod-storage.example"],"allowed_buckets":["prod-bucket"],"allowed_public_base_urls":["https://cdn.example.com"]}}
```

Development example:

```dotenv
ENVIRONMENT=development
BACKEND_REGISTRY_JSON={"backend-a-dev":{"environment":"development","service_key_sha256":"<sha256>","allowed_endpoint_urls":["https://dev-storage.example"],"allowed_buckets":["dev-bucket"],"allowed_public_base_urls":["https://dev-cdn.example.com"]}}
```

Generate each key against the matching environment file:

```bash
python scripts/generate_service_key.py \
  --backend-id backend-a-dev \
  --environment development \
  --env-file .env.development \
  --write
```

Store the printed `HEAR_SERVICE_KEY` only in the matching backend secret
store. The AI `.env` needs only the digest in `BACKEND_REGISTRY_JSON`.

Start a deployment with a selected environment file by exporting
`HEAR_ENV_FILE` before Supervisor or the server process starts:

```bash
HEAR_ENV_FILE=/workspace/hear-ai/.env.development \
  supervisord -c deploy/supervisord.conf
```

Production uses `/workspace/hear-ai/.env` by default. Keep the two files and
their PostgreSQL/storage credentials separate.
