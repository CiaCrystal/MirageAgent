---
id: exp-da081d10a5d1
target: pydantic-v2
versions:
  pydantic: '>=2.0,<3'
  observed: 2.14.1
error_signature: 'failure: ValidationError: string_type [pydantic.validation]'
error_type: failure
api_tags:
- ConfigDict
- allow_mutation
- class Config
- frozen=
- model_config
api_removed:
- allow_mutation
- class Config
api_added:
- ConfigDict
- frozen=
- model_config
preconditions:
- pydantic v2 is installed (source run used 2.14.1)
- the code still uses the v1 API `allow_mutation`
- the code still uses the v1 API `class Config`
- the failing traceback points into the module that defines the affected model/function
not_applicable:
- fixing it with `pydantic.v1` imports or by patching pydantic at runtime (not an acceptable migration)
- 'code that must keep running on pydantic v1 (the fix uses v2-only API: ConfigDict, frozen=, model_config)'
explanation_source: rule-template
combined: true
sources:
- repo: github.com/zuehlke/confz
  family: fam-confz
  commit: dfa9e97d050b82e715cea5a3bf0583c302074683
  run_id: source-001
  case: confz
  group: ''
  steps:
  - 13
  trace: ''
  evidence: ''
  files:
  - confz/confz.py
test_evidence:
- run_id: source-001
  step_before: 12
  step_after: 13
  passed_before: 37
  passed_after: 43
  total_after: 43
  signature_tests_before: 6
  resolved_tests:
  - tests.loaders.test_file_loader::test_yaml_file
  - tests.loaders.test_file_loader::test_multiple_yaml_files_both_available
  - tests.loaders.test_file_loader::test_multiple_yaml_files_one_is_optional_and_unavailable
  - tests.loaders.test_file_loader::test_custom_file
  - tests.loaders.test_file_loader::test_custom_file_str_path
  new_signatures_after: 0
  final_passed: 43
  final_total: 43
  signature_absent_at_end: true
status: approved
review:
  status: approved
  reason: Reviewed and approved for reuse.
  reviewer: maintainer
  reviewed_at: ''
  auto_checks:
    source_allowed: true
    run_strict_success: true
    run_no_compat_layer: true
    final_output_clean: true
    window_clean: true
    retained_in_final: true
    signature_absent_at_end: true
    tests_flipped_to_pass: true
    attribution_clear: true
    no_rollback_in_window: true
    edited_by_edit_tools: true
    diff_within_limit: true
    self_contained: true
    details:
      attribution: traceback frame
      window_files:
      - confz/confz.py
      window_tools:
      - edit_file
      window_issues: []
      final_output_issues: []
      source_policy: allowed
      diff_lines: 11
      hunks_in_files: 1
      hunks_kept: 1
      depends_on_earlier_definitions: {}
  auto_passed: true
  uncertain: []
content_hash: 12f5f741883d1955ff66ad381a4ace52c01c74fe99f257f560827864213d20d0
dedup_key: da081d10a5d146ff
reuse_success: 0
reuse_failure: 0
reuse_runs: []
created_at: ''
updated_at: ''
---

# failure: ValidationError: string_type [pydantic.validation]

## Error

```text
pydantic_core._pydantic_core.ValidationError: 1 validation error for OuterConfig
attr2
Input should be a valid string [type=string_type, input_value=2, input_type=int]
For further information visit https:<path>
```

## Code before

```python
    listeners: ClassVar[Optional[List[Any]]] = None  #: *for internal use only*

    class Config:
        allow_mutation = False

    @classmethod
```

## Fix (minimal diff)

```diff
--- a/confz/confz.py
+++ b/confz/confz.py
@@ -98,6 +98,8 @@
     listeners: ClassVar[Optional[List[Any]]] = None  #: *for internal use only*
 
-    class Config:
-        allow_mutation = False
+    # pydantic v1 coerced numbers to strings, v2 requires an explicit opt-in
+    model_config = ConfigDict(
+        frozen=True, coerce_numbers_to_str=True
+    )  #: pydantic model config
 
     @classmethod
```

## Explanation

In the source run the edit replaced `allow_mutation`, `class Config` with `ConfigDict`, `frozen=`, `model_config` in 1 file(s). Afterwards 6 test(s) that failed with this error no longer failed (suite 37/43 -> 43/43 passing). Rule-generated summary of the error -> edit -> test window; not written by a model.
