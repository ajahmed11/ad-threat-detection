"""Static guards on the Active Response script (no AD / PowerShell needed in CI).

These pin the fixes from the security review so they cannot silently regress.
"""
import re
from pathlib import Path

PS = Path("response/disable-ad-account.ps1").read_text()
XML = Path("response/ossec-manager-snippet.xml").read_text()
RULES = Path("detections/wazuh/local_rules.xml").read_text()


def test_machine_accounts_rejected_before_any_trim():
    assert "TrimEnd('$')" not in PS, "must not strip '$' before validating"
    assert "EndsWith('$')" in PS


def test_strict_sam_validation_and_single_object_resolution():
    assert "^[A-Za-z0-9._-]{1,64}$" in PS
    assert "$found.Count -ne 1" in PS
    # never pass raw alert text to the AD cmdlets
    assert not re.search(r"Disable-ADAccount\s+-Identity\s+\$user\b", PS)


def test_rule_allowlist_inside_script_and_only_dcsync_acts():
    assert "$RuleActions" in PS
    acting = re.findall(r'"(\d+)"\s*=\s*"[A-Za-z]+"', PS)
    assert acting == ["100514"], acting


def test_manager_binds_only_dcsync():
    ids = re.search(r"<rules_id>([^<]+)</rules_id>", XML).group(1).split(",")
    assert ids == ["100514"]


def test_protected_identity_checks_present():
    for needle in ("adminCount", "RID", "LDAP", "1.2.840.113556.1.4.1941"):
        assert needle in PS or needle.lower() in PS.lower(), needle


def test_dry_run_is_default():
    assert "enforce.flag" in PS and "DRY-RUN" in PS


def test_dcsync_rule_excludes_machine_accounts():
    block = RULES[RULES.index('id="100514"'):]
    assert 'negate="yes"' in block


def test_spray_parent_not_audit_success_only():
    block = RULES[RULES.index('id="100513"'):RULES.index('id="100515"')]
    assert "60104" in block or "60105" in block
