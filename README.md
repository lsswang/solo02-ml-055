# Censored count demand and perishable ordering

Run `python demo.py` after installing `requirements.txt`.

The current demand module fits a real independent truncated-count model to fully observed counts, with a two-parameter ridge objective and analytic gradient. Its forecast is an independent complete path table. The inventory module solves a real one-period immediate-delivery newsvendor problem. The demo uses the fitted distribution to place an actual order.

`fixtures/cases.json` also contains a short censored sales example and a three-period delayed perishable ordering request for the extension. Historical sales are `min(actual demand, cap)`. Future order requests, including requests that cannot be fulfilled, are observable after that period's order. The intended extension contract is specified in the task text.
