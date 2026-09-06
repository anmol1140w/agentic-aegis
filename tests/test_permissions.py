from runtime.permissions import PermissionEffect, PermissionManager, PermissionRule


def test_permission_precedence_and_session_scope() -> None:
    manager = PermissionManager([
        PermissionRule("filesystem.write", "workspace/*.py", PermissionEffect.ALLOW, "s1"),
        PermissionRule("filesystem.write", "workspace/secrets.py", PermissionEffect.DENY, "s1"),
    ])
    assert manager.evaluate("s1", "filesystem.write", ["workspace/main.py"]) is PermissionEffect.ALLOW
    assert manager.evaluate("s1", "filesystem.write", ["workspace/secrets.py"]) is PermissionEffect.DENY
    assert manager.evaluate("s2", "filesystem.write", ["workspace/main.py"]) is PermissionEffect.PROMPT


def test_permission_request_can_be_saved() -> None:
    manager = PermissionManager()
    request = manager.request("s1", "terminal.execute", ["pytest"])
    assert request.status == "pending"
    manager.reply(request.request_id, True, save=True)
    assert manager.evaluate("s1", "terminal.execute", ["pytest"]) is PermissionEffect.ALLOW
