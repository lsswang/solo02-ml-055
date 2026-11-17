import json
from pathlib import Path
import demand
import inventory


def run():
    fixture = json.loads((Path(__file__).parent / "fixtures/cases.json").read_text())
    fit = demand.fit(**fixture["ordinary_fit"])
    table = demand.forecast(fit["theta"], [0.0])
    decision = inventory.choose_order([row["probability"] for row in table], **fixture["ordinary_inventory"])
    return {"fit": fit, "forecast": table, "decision": decision}


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
