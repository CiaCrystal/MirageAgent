---
description: "BaseSettings moved to the pydantic-settings package; settings config keys"
---
# Settings

- `from pydantic import BaseSettings` -> `from pydantic_settings import BaseSettings` (separate package
  `pydantic-settings`; install it if it is missing).
- `class Config: env_prefix = ...` -> `model_config = SettingsConfigDict(env_prefix=...)`.
