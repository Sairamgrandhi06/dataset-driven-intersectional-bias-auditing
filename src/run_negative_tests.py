"""
Negative Test Campaign Runner Script (NT-1 through NT-5).

Executes controlled negative tests NT-1, NT-2, NT-3, NT-4, and NT-5 against synthetic fixtures.
Asserts failure detection, safe response, recovery execution, reconciliation, and evidence retention.
Exits 0 if all tests pass, 1 if any fail.
"""

import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.reliability.recovery_manager import RecoveryManager


def run_negative_test_campaign():
    print("=" * 80)
    print("PHASE 11 NEGATIVE TEST CAMPAIGN")
    print("=" * 80)

    mgr = RecoveryManager()

    # NT-1
    res1 = mgr.run_nt1_recovery_workflow()
    st1 = "DETECTED -> RECOVERED" if res1["status"] == "RECOVERED" else "FAILED"
    print(f"NT-1 Missing intersection ........ {st1}")

    # NT-2
    res2 = mgr.run_nt2_recovery_workflow()
    st2 = "DETECTED -> RECOVERED" if res2["status"] == "RECOVERED" else "FAILED"
    print(f"NT-2 Hidden mitigation cost ..... {st2}")

    # NT-3
    res3 = mgr.run_nt3_recovery_workflow()
    st3 = "DETECTED -> RECOVERED" if res3["status"] == "RECOVERED" else "FAILED"
    print(f"NT-3 Silent criterion selection . {st3}")

    # NT-4
    res4 = mgr.run_nt4_recovery_workflow()
    st4 = "BLOCKED -> RECOVERED" if res4["status"] == "RECOVERED" else "FAILED"
    print(f"NT-4 Sensitive evidence ........ {st4}")

    # NT-5
    res5 = mgr.run_nt5_recovery_workflow()
    st5 = "REJECTED -> RECOVERED" if res5["status"] == "RECOVERED" else "FAILED"
    print(f"NT-5 Schema evolution .......... {st5}")

    all_recovered = all(r["status"] == "RECOVERED" for r in [res1, res2, res3, res4, res5])
    all_recon = all(r["reconciliation"] == "PASS" for r in [res1, res2, res3, res4, res5])

    print("\nRecovery evidence:")
    print("PASS" if all_recovered else "FAIL")

    print("\nReconciliation:")
    print("PASS" if all_recon else "FAIL")

    print("\n" + "-" * 80)
    campaign_pass = all_recovered and all_recon
    print(f"Phase 11 negative-test status:\n{'PASS' if campaign_pass else 'FAIL'}")
    print("=" * 80)

    if not campaign_pass:
        sys.exit(1)

    return True


if __name__ == "__main__":
    run_negative_test_campaign()
