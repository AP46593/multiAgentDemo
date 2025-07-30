from __future__ import annotations
from typing import Dict, Optional, cast

import os, json, re, requests
from schema import GraphMessage
from langchain_core.runnables import RunnableConfig

# ─────────────  constants  ──────────────
PLATE_RE = re.compile(
    r"""^(
        [A-Z]{2}[0-9]{2}[A-Z]{2,3}          # UK current (AA99AA / AA99AAA)
        |                                   # ──or──
        [A-Z]{1}[0-9]{1,3}[A-Z]{3}          # UK prefix (A123AAA)
        |                                   # ──or──
        [0-9]{2}BH[0-9]{4}[A-Z]?            # India BH series (22BH6517A)
        |                                   # ──or──
        [A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{1,4}  # India general (KA01AB1234)
        |                                   # ──or──
        [A-Z]{3,4}[0-9]{2,4}                # Generic fallback (PFZI260, ABC123)
    )$""",
    re.VERBOSE,
)     
DVLA_URL = "https://driver-vehicle-licensing.api.gov.uk/vehicle-enquiry/v1/vehicles"
API_KEY  = os.getenv("DVLA_API_KEY", "")

# ─────────────  dummy response for demo  ──────────────
DEMO_VEHICLE = {
    "registrationNumber": "SP05WFM",
    "taxStatus": "SORN",
    "motStatus": "Not valid",
    "make": "BMW",
    "yearOfManufacture": 2005,
    "engineCapacity": 1596,
    "co2Emissions": 181,
    "fuelType": "PETROL",
    "markedForExport": False,
    "colour": "BLUE",
    "typeApproval": "M1",
    "dateOfLastV5CIssued": "2024-01-02",
    "motExpiryDate": "2024-11-26",
    "wheelplan": "2 AXLE RIGID BODY",
    "monthOfFirstRegistration": "2005-03",
}

# ─────────────  agent  ──────────────
def processing_agent(state: GraphMessage,
               config: RunnableConfig | None = None) -> GraphMessage:
    msg = state

    # 1️⃣ Ensure we actually got a plate to work with
    if msg.data is None or "plate" not in msg.data:
        return GraphMessage(
            role="assistant",
            text="⚠️ No licence‑plate to process. Please try again."
        )

    plate: str = cast(str, msg.data["plate"])

    # 2️⃣ Validate plate format
    if not PLATE_RE.match(plate):
        return GraphMessage(
            role="assistant",
            text=f"❌ '{plate}' doesn’t look like a valid registration number."
        )

    # 3️⃣ Check we have an API key
    if not API_KEY:
        vehicle = {**DEMO_VEHICLE, "registrationNumber": plate}  # ← our hard‑coded dict
        return GraphMessage(
            role="delegate",
            text="chat",
            data={"vehicle": vehicle},
        )

    # 4️⃣  Call DVLA Vehicle‑Enquiry API (only reached if key present)
    headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
    payload = json.dumps({"registrationNumber": plate})

    try:
        res = requests.post(DVLA_URL, headers=headers, data=payload, timeout=10)
        res.raise_for_status()
        vehicle = res.json()

        return GraphMessage(
            role="delegate",
            text="chat",
            data={"vehicle": vehicle},
        )

    except requests.RequestException as err:
        return GraphMessage(
            role="assistant",
            text=f"DVLA lookup failed: {err}",
        )
    return GraphMessage(
        role="assistant",
        text="⚠️ Unexpected flow; no response generated."
    )
