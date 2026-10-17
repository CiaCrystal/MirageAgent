---
description: "custom types (__get_validators__), GenericModel, validate_arguments"
---
# Custom types and other APIs

- Custom type `__get_validators__` -> `__get_pydantic_core_schema__`, or `Annotated[T, BeforeValidator(...)]` /
  `AfterValidator(...)`.
- `GenericModel` -> plain `BaseModel` + `Generic[T]`.
- `pydantic.validate_arguments` -> `pydantic.validate_call`.
