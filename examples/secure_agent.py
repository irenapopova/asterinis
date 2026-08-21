from asterinis.security import (
    ExecutionCounter,
    ExecutionLimits,
    PermissionPolicy,
    validate_input,
)


def main() -> None:
    text = validate_input(
        "Search trusted documents."
    )

    permissions = PermissionPolicy()
    permissions.allow_tool("search")

    limits = ExecutionLimits(
        max_agent_steps=5,
        max_tool_calls=3,
        max_retries=1,
    )

    counter = ExecutionCounter(limits)

    counter.record_agent_step()

    if permissions.can_use_tool("search"):
        counter.record_tool_call()
        print(f"Allowed request: {text}")
    else:
        print("Search tool is not permitted.")


if __name__ == "__main__":
    main()