"""CI: detections are valid and well-formed (no lab needed)."""
import glob
import xml.etree.ElementTree as ET
import yaml

REQUIRED_SIGMA = {"title", "logsource", "detection", "level"}


def test_sigma_rules_parse_and_have_required_fields():
    files = glob.glob("detections/sigma/*.yml")
    assert files, "no sigma rules found"
    for f in files:
        doc = yaml.safe_load(open(f))
        missing = REQUIRED_SIGMA - set(doc)
        assert not missing, f"{f} missing {missing}"
        assert "condition" in doc["detection"], f"{f} detection has no condition"


def test_sigma_rules_are_attack_mapped():
    for f in glob.glob("detections/sigma/*.yml"):
        doc = yaml.safe_load(open(f))
        tags = doc.get("tags", [])
        assert any(t.startswith("attack.t") for t in tags), f"{f} not ATT&CK-tagged"


def test_wazuh_local_rules_well_formed():
    # the file is an XML fragment; wrap it so ElementTree can parse the group(s)
    raw = open("detections/wazuh/local_rules.xml").read()
    ET.fromstring("<root>" + raw + "</root>")


def test_attack_navigator_layer_valid():
    import json
    layer = json.load(open("attack/ad-detection-coverage.navigator.json"))
    assert layer["domain"] == "enterprise-attack"
    assert layer["techniques"], "no techniques in layer"
    for t in layer["techniques"]:
        assert t["techniqueID"].startswith("T"), t
