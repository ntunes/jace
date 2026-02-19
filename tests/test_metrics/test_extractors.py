"""Tests for metric extractors with realistic Junos output samples."""

from __future__ import annotations

from jace.device.models import CommandResult
from jace.metrics.routing import extract_routing_metrics
from jace.metrics.chassis import extract_chassis_metrics
from jace.metrics.interfaces import extract_interface_metrics
from jace.metrics.system import extract_system_metrics


# ── Routing ──────────────────────────────────────────────────────────

ROUTE_SUMMARY_OUTPUT = """\
Router ID: 10.0.0.1

inet.0: 15 destinations, 20 routes (15 active, 0 holddown, 0 hidden)
              Direct:      5 routes,      5 active
              Local:       5 routes,      5 active
              BGP:         8 routes,      4 active
              OSPF:        2 routes,      1 active

inet6.0: 3 destinations, 3 routes (3 active, 0 holddown, 0 hidden)
"""

BGP_SUMMARY_OUTPUT = """\
Threading mode: BGP I/O
Groups: 2 Peers: 3 Down peers: 1
Table          Tot Paths  Act Paths Suppressed    History Damp State    Pending
inet.0                8          4          0          0          0          0
Peer                     AS      InPkt     OutPkt    OutQ   Flaps Last Up/Dwn State|#Active/Received/Accepted/Damped...
10.0.0.2              64512        100        101       0       1     1:23:45 4/8/8/0              0/0/0/0
10.0.0.3              64513         50         51       0       0     0:45:12 0/0/0/0              0/0/0/0
10.0.0.4              64514          0          0       0       2     0:00:30 Active
"""

OSPF_NEIGHBOR_OUTPUT = """\
Address          Interface              State     ID               Pri  Dead
10.0.1.2         ge-0/0/0.0             Full      10.0.0.2         128    37
10.0.1.3         ge-0/0/1.0             Full      10.0.0.3         128    32
"""


def test_extract_routing_metrics_route_summary():
    results = {
        "show route summary": CommandResult(
            command="show route summary", output=ROUTE_SUMMARY_OUTPUT,
        ),
    }
    metrics = extract_routing_metrics(results)
    by_name = {m.metric: m for m in metrics}

    assert "route_total" in by_name
    assert by_name["route_total"].value == 20.0
    assert by_name["route_total"].unit == "routes"

    assert "route_active" in by_name
    assert by_name["route_active"].value == 15.0


def test_extract_routing_metrics_bgp():
    results = {
        "show bgp summary": CommandResult(
            command="show bgp summary", output=BGP_SUMMARY_OUTPUT,
        ),
    }
    metrics = extract_routing_metrics(results)
    by_name = {m.metric: m for m in metrics}

    assert by_name["bgp_peer_count"].value == 3.0
    assert by_name["bgp_established_count"].value == 2.0  # two have prefix counts


def test_extract_routing_metrics_ospf():
    results = {
        "show ospf neighbor": CommandResult(
            command="show ospf neighbor", output=OSPF_NEIGHBOR_OUTPUT,
        ),
    }
    metrics = extract_routing_metrics(results)
    by_name = {m.metric: m for m in metrics}

    assert by_name["ospf_neighbor_count"].value == 2.0


def test_extract_routing_metrics_empty():
    metrics = extract_routing_metrics({})
    assert metrics == []


def test_extract_routing_metrics_failed_command():
    results = {
        "show route summary": CommandResult(
            command="show route summary", output="", success=False,
            error="connection reset",
        ),
    }
    metrics = extract_routing_metrics(results)
    assert metrics == []


# ── Chassis ──────────────────────────────────────────────────────────

ROUTING_ENGINE_OUTPUT = """\
Routing Engine status:
  Slot 0:
    Current state                  Master
    Election priority              Master (default)
    Temperature                 45 degrees C / 113 degrees F
    CPU utilization:
      User                         12 percent
      Background                    0 percent
      Kernel                        5 percent
      Interrupt                     1 percent
      Idle                         82 percent
    CPU utilization:              18 percent
    Memory utilization          62 percent
    Model                          RE-S-1800x4
    Start time                     2024-01-15 08:30:00 UTC
    Uptime                         45 days, 12:34:56
    Load averages:   1 minute   5 minute  15 minute
                        0.25      0.18      0.15
"""

PFE_EXCEPTIONS_OUTPUT = """\
PFE statistics:
  Memory allocation errors:     0
  Bad route lookups:            5
  TTL expired:                 12
  Packets punted:               0
  Hardware errors:              3
"""


def test_extract_chassis_metrics_re():
    results = {
        "show chassis routing-engine": CommandResult(
            command="show chassis routing-engine",
            output=ROUTING_ENGINE_OUTPUT,
        ),
    }
    metrics = extract_chassis_metrics(results)
    by_name = {m.metric: m for m in metrics}

    assert "re_cpu_pct" in by_name
    assert by_name["re_cpu_pct"].value == 18.0
    assert by_name["re_cpu_pct"].unit == "%"

    assert "re_memory_pct" in by_name
    assert by_name["re_memory_pct"].value == 62.0


def test_extract_chassis_metrics_pfe():
    results = {
        "show pfe statistics exceptions": CommandResult(
            command="show pfe statistics exceptions",
            output=PFE_EXCEPTIONS_OUTPUT,
        ),
    }
    metrics = extract_chassis_metrics(results)
    by_name = {m.metric: m for m in metrics}

    # Only non-zero counters
    assert "pfe_exception_bad_route_lookups" in by_name
    assert by_name["pfe_exception_bad_route_lookups"].value == 5.0
    assert by_name["pfe_exception_bad_route_lookups"].is_counter is True

    assert "pfe_exception_ttl_expired" in by_name
    assert by_name["pfe_exception_ttl_expired"].value == 12.0

    assert "pfe_exception_hardware_errors" in by_name
    assert by_name["pfe_exception_hardware_errors"].value == 3.0

    # Zero values should not appear
    assert "pfe_exception_memory_allocation_errors" not in by_name
    assert "pfe_exception_packets_punted" not in by_name


def test_extract_chassis_metrics_empty():
    metrics = extract_chassis_metrics({})
    assert metrics == []


# ── Interfaces ───────────────────────────────────────────────────────

INTERFACES_TERSE_OUTPUT = """\
Interface               Admin Link Proto    Local                 Remote
ge-0/0/0                up    up
ge-0/0/0.0              up    up   inet     10.0.1.1/24
ge-0/0/1                up    down
ge-0/0/2                up    up
ge-0/0/3                down  down
lo0                     up    up
lo0.0                   up    up   inet     10.0.0.1/32
"""

INTERFACES_STATS_OUTPUT = """\
Physical interface: ge-0/0/0, Enabled, Physical link is Up
  Input errors:                5, Input drops: 0
  Output errors:               2, Output drops: 0

Physical interface: ge-0/0/1, Enabled, Physical link is Down
  Input errors:                0, Input drops: 0
  Output errors:               0, Output drops: 0
"""


def test_extract_interface_metrics_terse():
    results = {
        "show interfaces terse": CommandResult(
            command="show interfaces terse",
            output=INTERFACES_TERSE_OUTPUT,
        ),
    }
    metrics = extract_interface_metrics(results)
    by_name = {m.metric: m for m in metrics}

    assert by_name["iface_up_count"].value == 3.0  # ge-0/0/0, ge-0/0/2, lo0
    assert by_name["iface_down_count"].value == 1.0  # ge-0/0/1 (admin up, link down)


def test_extract_interface_metrics_errors():
    results = {
        "show interfaces statistics": CommandResult(
            command="show interfaces statistics",
            output=INTERFACES_STATS_OUTPUT,
        ),
    }
    metrics = extract_interface_metrics(results)
    by_name = {m.metric: m for m in metrics}

    assert by_name["iface_error_count"].value == 7.0  # 5 + 2
    assert by_name["iface_error_count"].is_counter is True


def test_extract_interface_metrics_empty():
    metrics = extract_interface_metrics({})
    assert metrics == []


# ── System ───────────────────────────────────────────────────────────

SYSTEM_STORAGE_OUTPUT = """\
Filesystem              Size       Used      Avail  Capacity   Mounted on
/dev/gpt/junos          5.8G       2.1G       3.3G       39%   /
devfs                   1.0K       1.0K         0B      100%   /dev
/dev/md0                185M       185M         0B      100%   /packages/mnt/jbase
tmpfs                   2.0G        12M       1.9G        1%   /tmp
/dev/gpt/config         396M        56K       364M        1%   /config
"""


def test_extract_system_metrics_storage():
    results = {
        "show system storage": CommandResult(
            command="show system storage",
            output=SYSTEM_STORAGE_OUTPUT,
        ),
    }
    metrics = extract_system_metrics(results)
    by_name = {m.metric: m for m in metrics}

    assert "disk_used_pct" in by_name
    assert by_name["disk_used_pct"].value == 100.0  # highest is 100%


def test_extract_system_metrics_load_avg():
    results = {
        "show chassis routing-engine": CommandResult(
            command="show chassis routing-engine",
            output=ROUTING_ENGINE_OUTPUT,
        ),
    }
    metrics = extract_system_metrics(results)
    by_name = {m.metric: m for m in metrics}

    assert "re_load_avg" in by_name
    assert by_name["re_load_avg"].value == 0.25


def test_extract_system_metrics_empty():
    metrics = extract_system_metrics({})
    assert metrics == []
