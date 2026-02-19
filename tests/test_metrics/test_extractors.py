"""Tests for metric extractors — both XML (PyEZ) and text (Netmiko) paths."""

from __future__ import annotations

from xml.etree.ElementTree import Element, SubElement, fromstring

from jace.device.models import CommandResult
from jace.metrics.routing import extract_routing_metrics
from jace.metrics.chassis import extract_chassis_metrics
from jace.metrics.interfaces import extract_interface_metrics
from jace.metrics.system import extract_system_metrics


# ═══════════════════════════════════════════════════════════════════════
# XML test data (simulates PyEZ RPC output — MX series)
# ═══════════════════════════════════════════════════════════════════════

ROUTE_SUMMARY_XML = fromstring("""\
<route-summary-information>
  <route-table>
    <table-name>inet.0</table-name>
    <destination-count>15</destination-count>
    <total-route-count>20</total-route-count>
    <active-route-count>15</active-route-count>
    <holddown-route-count>0</holddown-route-count>
    <hidden-route-count>0</hidden-route-count>
  </route-table>
  <route-table>
    <table-name>inet6.0</table-name>
    <destination-count>3</destination-count>
    <total-route-count>3</total-route-count>
    <active-route-count>3</active-route-count>
  </route-table>
</route-summary-information>
""")

BGP_SUMMARY_XML = fromstring("""\
<bgp-information>
  <group-count>2</group-count>
  <peer-count>3</peer-count>
  <down-peer-count>1</down-peer-count>
  <bgp-peer>
    <peer-address>10.0.0.2</peer-address>
    <peer-as>64512</peer-as>
    <peer-state>Established</peer-state>
  </bgp-peer>
  <bgp-peer>
    <peer-address>10.0.0.3</peer-address>
    <peer-as>64513</peer-as>
    <peer-state>Established</peer-state>
  </bgp-peer>
  <bgp-peer>
    <peer-address>10.0.0.4</peer-address>
    <peer-as>64514</peer-as>
    <peer-state>Active</peer-state>
  </bgp-peer>
</bgp-information>
""")

OSPF_NEIGHBOR_XML = fromstring("""\
<ospf-neighbor-information>
  <ospf-neighbor>
    <neighbor-address>10.0.1.2</neighbor-address>
    <interface-name>ge-0/0/0.0</interface-name>
    <ospf-neighbor-state>Full</ospf-neighbor-state>
    <neighbor-id>10.0.0.2</neighbor-id>
  </ospf-neighbor>
  <ospf-neighbor>
    <neighbor-address>10.0.1.3</neighbor-address>
    <interface-name>ge-0/0/1.0</interface-name>
    <ospf-neighbor-state>Full</ospf-neighbor-state>
    <neighbor-id>10.0.0.3</neighbor-id>
  </ospf-neighbor>
</ospf-neighbor-information>
""")

ROUTING_ENGINE_XML = fromstring("""\
<route-engine-information>
  <route-engine>
    <slot>0</slot>
    <mastership-state>master</mastership-state>
    <cpu-user>12</cpu-user>
    <cpu-background>0</cpu-background>
    <cpu-system>5</cpu-system>
    <cpu-interrupt>1</cpu-interrupt>
    <cpu-idle>82</cpu-idle>
    <memory-buffer-utilization>62</memory-buffer-utilization>
    <load-average-one>0.25</load-average-one>
    <load-average-five>0.18</load-average-five>
    <load-average-fifteen>0.15</load-average-fifteen>
  </route-engine>
</route-engine-information>
""")

PFE_STATS_XML = fromstring("""\
<pfe-statistics>
  <pfe-hardware-discard-statistics>
    <timeout-discard>0</timeout-discard>
    <bad-route-discard>5</bad-route-discard>
    <nexthop-discard>12</nexthop-discard>
    <invalid-iif-discard>0</invalid-iif-discard>
    <data-error-discard>3</data-error-discard>
    <stack-underflow-discard>0</stack-underflow-discard>
    <stack-overflow-discard>0</stack-overflow-discard>
    <truncated-key-discard>0</truncated-key-discard>
    <bits-to-test-discard>0</bits-to-test-discard>
    <info-cell-discard>0</info-cell-discard>
    <fabric-discard>0</fabric-discard>
  </pfe-hardware-discard-statistics>
</pfe-statistics>
""")

INTERFACES_TERSE_XML = fromstring("""\
<interface-information>
  <physical-interface>
    <name>ge-0/0/0</name>
    <admin-status>up</admin-status>
    <oper-status>up</oper-status>
  </physical-interface>
  <physical-interface>
    <name>ge-0/0/1</name>
    <admin-status>up</admin-status>
    <oper-status>down</oper-status>
  </physical-interface>
  <physical-interface>
    <name>ge-0/0/2</name>
    <admin-status>up</admin-status>
    <oper-status>up</oper-status>
  </physical-interface>
  <physical-interface>
    <name>ge-0/0/3</name>
    <admin-status>down</admin-status>
    <oper-status>down</oper-status>
  </physical-interface>
  <physical-interface>
    <name>lo0</name>
    <admin-status>up</admin-status>
    <oper-status>up</oper-status>
  </physical-interface>
</interface-information>
""")

INTERFACES_STATS_XML = fromstring("""\
<interface-information>
  <physical-interface>
    <name>ge-0/0/0</name>
    <input-error-list>
      <input-errors>5</input-errors>
      <input-drops>0</input-drops>
    </input-error-list>
    <output-error-list>
      <output-errors>2</output-errors>
      <output-drops>0</output-drops>
    </output-error-list>
  </physical-interface>
  <physical-interface>
    <name>ge-0/0/1</name>
    <input-error-list>
      <input-errors>0</input-errors>
      <input-drops>0</input-drops>
    </input-error-list>
    <output-error-list>
      <output-errors>0</output-errors>
      <output-drops>0</output-drops>
    </output-error-list>
  </physical-interface>
</interface-information>
""")

SYSTEM_STORAGE_XML = fromstring("""\
<system-storage-information>
  <filesystem>
    <filesystem-name>/dev/gpt/junos</filesystem-name>
    <used-percent>39</used-percent>
    <mounted-on>/</mounted-on>
  </filesystem>
  <filesystem>
    <filesystem-name>/dev/md0.uzip</filesystem-name>
    <used-percent>100</used-percent>
    <mounted-on>/packages/mnt/jbase</mounted-on>
  </filesystem>
  <filesystem>
    <filesystem-name>tmpfs</filesystem-name>
    <used-percent>1</used-percent>
    <mounted-on>/tmp</mounted-on>
  </filesystem>
</system-storage-information>
""")


# ═══════════════════════════════════════════════════════════════════════
# Text test data (simulates Netmiko CLI output)
# ═══════════════════════════════════════════════════════════════════════

ROUTE_SUMMARY_TEXT = """\
Router ID: 10.0.0.1

inet.0: 15 destinations, 20 routes (15 active, 0 holddown, 0 hidden)
              Direct:      5 routes,      5 active
              Local:       5 routes,      5 active
              BGP:         8 routes,      4 active
              OSPF:        2 routes,      1 active

inet6.0: 3 destinations, 3 routes (3 active, 0 holddown, 0 hidden)
"""

BGP_SUMMARY_TEXT = """\
Threading mode: BGP I/O
Groups: 2 Peers: 3 Down peers: 1
Table          Tot Paths  Act Paths Suppressed    History Damp State    Pending
inet.0                8          4          0          0          0          0
Peer                     AS      InPkt     OutPkt    OutQ   Flaps Last Up/Dwn State|#Active/Received/Accepted/Damped...
10.0.0.2              64512        100        101       0       1     1:23:45 4/8/8/0              0/0/0/0
10.0.0.3              64513         50         51       0       0     0:45:12 0/0/0/0              0/0/0/0
10.0.0.4              64514          0          0       0       2     0:00:30 Active
"""

OSPF_NEIGHBOR_TEXT = """\
Address          Interface              State     ID               Pri  Dead
10.0.1.2         ge-0/0/0.0             Full      10.0.0.2         128    37
10.0.1.3         ge-0/0/1.0             Full      10.0.0.3         128    32
"""

ROUTING_ENGINE_TEXT = """\
Routing Engine status:
  Slot 0:
    Current state                  Master
    CPU utilization:              18 percent
    Memory utilization          62 percent
    Load averages:   1 minute   5 minute  15 minute
                        0.25      0.18      0.15
"""

PFE_EXCEPTIONS_TEXT = """\
PFE statistics:
  Memory allocation errors:     0
  Bad route lookups:            5
  TTL expired:                 12
  Packets punted:               0
  Hardware errors:              3
"""

INTERFACES_TERSE_TEXT = """\
Interface               Admin Link Proto    Local                 Remote
ge-0/0/0                up    up
ge-0/0/0.0              up    up   inet     10.0.1.1/24
ge-0/0/1                up    down
ge-0/0/2                up    up
ge-0/0/3                down  down
lo0                     up    up
lo0.0                   up    up   inet     10.0.0.1/32
"""

INTERFACES_STATS_TEXT = """\
Physical interface: ge-0/0/0, Enabled, Physical link is Up
  Input errors:                5, Input drops: 0
  Output errors:               2, Output drops: 0

Physical interface: ge-0/0/1, Enabled, Physical link is Down
  Input errors:                0, Input drops: 0
  Output errors:               0, Output drops: 0
"""

SYSTEM_STORAGE_TEXT = """\
Filesystem              Size       Used      Avail  Capacity   Mounted on
/dev/gpt/junos          5.8G       2.1G       3.3G       39%   /
devfs                   1.0K       1.0K         0B      100%   /dev
/dev/md0                185M       185M         0B      100%   /packages/mnt/jbase
tmpfs                   2.0G        12M       1.9G        1%   /tmp
/dev/gpt/config         396M        56K       364M        1%   /config
"""


# ═══════════════════════════════════════════════════════════════════════
# Routing — XML path
# ═══════════════════════════════════════════════════════════════════════

def test_routing_xml_route_summary():
    results = {
        "show route summary": CommandResult(
            command="show route summary", output="",
            structured=ROUTE_SUMMARY_XML,
        ),
    }
    metrics = extract_routing_metrics(results)
    by_name = {m.metric: m for m in metrics}

    assert by_name["route_total"].value == 23.0  # 20 + 3
    assert by_name["route_total"].unit == "routes"
    assert by_name["route_active"].value == 18.0  # 15 + 3


def test_routing_xml_bgp():
    results = {
        "show bgp summary": CommandResult(
            command="show bgp summary", output="",
            structured=BGP_SUMMARY_XML,
        ),
    }
    metrics = extract_routing_metrics(results)
    by_name = {m.metric: m for m in metrics}

    assert by_name["bgp_peer_count"].value == 3.0
    assert by_name["bgp_established_count"].value == 2.0


def test_routing_xml_ospf():
    results = {
        "show ospf neighbor": CommandResult(
            command="show ospf neighbor", output="",
            structured=OSPF_NEIGHBOR_XML,
        ),
    }
    metrics = extract_routing_metrics(results)
    by_name = {m.metric: m for m in metrics}

    assert by_name["ospf_neighbor_count"].value == 2.0


# ═══════════════════════════════════════════════════════════════════════
# Routing — text fallback
# ═══════════════════════════════════════════════════════════════════════

def test_routing_text_route_summary():
    results = {
        "show route summary": CommandResult(
            command="show route summary", output=ROUTE_SUMMARY_TEXT,
        ),
    }
    metrics = extract_routing_metrics(results)
    by_name = {m.metric: m for m in metrics}

    assert by_name["route_total"].value == 20.0
    assert by_name["route_active"].value == 15.0


def test_routing_text_bgp():
    results = {
        "show bgp summary": CommandResult(
            command="show bgp summary", output=BGP_SUMMARY_TEXT,
        ),
    }
    metrics = extract_routing_metrics(results)
    by_name = {m.metric: m for m in metrics}

    assert by_name["bgp_peer_count"].value == 3.0
    assert by_name["bgp_established_count"].value == 2.0


def test_routing_text_ospf():
    results = {
        "show ospf neighbor": CommandResult(
            command="show ospf neighbor", output=OSPF_NEIGHBOR_TEXT,
        ),
    }
    metrics = extract_routing_metrics(results)
    by_name = {m.metric: m for m in metrics}

    assert by_name["ospf_neighbor_count"].value == 2.0


def test_routing_empty():
    assert extract_routing_metrics({}) == []


def test_routing_failed_command():
    results = {
        "show route summary": CommandResult(
            command="show route summary", output="", success=False,
            error="connection reset",
        ),
    }
    assert extract_routing_metrics(results) == []


# ═══════════════════════════════════════════════════════════════════════
# Chassis — XML path
# ═══════════════════════════════════════════════════════════════════════

def test_chassis_xml_re():
    results = {
        "show chassis routing-engine": CommandResult(
            command="show chassis routing-engine", output="",
            structured=ROUTING_ENGINE_XML,
        ),
    }
    metrics = extract_chassis_metrics(results)
    by_name = {m.metric: m for m in metrics}

    assert by_name["re_cpu_pct"].value == 18.0  # 100 - 82 idle
    assert by_name["re_cpu_pct"].unit == "%"
    assert by_name["re_memory_pct"].value == 62.0


def test_chassis_xml_pfe():
    results = {
        "show pfe statistics exceptions": CommandResult(
            command="show pfe statistics exceptions", output="",
            structured=PFE_STATS_XML,
        ),
    }
    metrics = extract_chassis_metrics(results)
    by_name = {m.metric: m for m in metrics}

    assert by_name["pfe_exception_bad_route_discard"].value == 5.0
    assert by_name["pfe_exception_bad_route_discard"].is_counter is True
    assert by_name["pfe_exception_nexthop_discard"].value == 12.0
    assert by_name["pfe_exception_data_error_discard"].value == 3.0
    # Zero values should not appear
    assert "pfe_exception_timeout_discard" not in by_name


# ═══════════════════════════════════════════════════════════════════════
# Chassis — text fallback
# ═══════════════════════════════════════════════════════════════════════

def test_chassis_text_re():
    results = {
        "show chassis routing-engine": CommandResult(
            command="show chassis routing-engine",
            output=ROUTING_ENGINE_TEXT,
        ),
    }
    metrics = extract_chassis_metrics(results)
    by_name = {m.metric: m for m in metrics}

    assert by_name["re_cpu_pct"].value == 18.0
    assert by_name["re_memory_pct"].value == 62.0


def test_chassis_text_pfe():
    results = {
        "show pfe statistics exceptions": CommandResult(
            command="show pfe statistics exceptions",
            output=PFE_EXCEPTIONS_TEXT,
        ),
    }
    metrics = extract_chassis_metrics(results)
    by_name = {m.metric: m for m in metrics}

    assert "pfe_exception_bad_route_lookups" in by_name
    assert by_name["pfe_exception_bad_route_lookups"].value == 5.0
    assert by_name["pfe_exception_ttl_expired"].value == 12.0
    assert by_name["pfe_exception_hardware_errors"].value == 3.0
    assert "pfe_exception_memory_allocation_errors" not in by_name


def test_chassis_empty():
    assert extract_chassis_metrics({}) == []


# ═══════════════════════════════════════════════════════════════════════
# Interfaces — XML path
# ═══════════════════════════════════════════════════════════════════════

def test_interfaces_xml_terse():
    results = {
        "show interfaces terse": CommandResult(
            command="show interfaces terse", output="",
            structured=INTERFACES_TERSE_XML,
        ),
    }
    metrics = extract_interface_metrics(results)
    by_name = {m.metric: m for m in metrics}

    assert by_name["iface_up_count"].value == 3.0  # ge-0/0/0, ge-0/0/2, lo0
    assert by_name["iface_down_count"].value == 1.0  # ge-0/0/1


def test_interfaces_xml_errors():
    results = {
        "show interfaces statistics": CommandResult(
            command="show interfaces statistics", output="",
            structured=INTERFACES_STATS_XML,
        ),
    }
    metrics = extract_interface_metrics(results)
    by_name = {m.metric: m for m in metrics}

    assert by_name["iface_error_count"].value == 7.0  # 5 + 2
    assert by_name["iface_error_count"].is_counter is True


# ═══════════════════════════════════════════════════════════════════════
# Interfaces — text fallback
# ═══════════════════════════════════════════════════════════════════════

def test_interfaces_text_terse():
    results = {
        "show interfaces terse": CommandResult(
            command="show interfaces terse",
            output=INTERFACES_TERSE_TEXT,
        ),
    }
    metrics = extract_interface_metrics(results)
    by_name = {m.metric: m for m in metrics}

    assert by_name["iface_up_count"].value == 3.0
    assert by_name["iface_down_count"].value == 1.0


def test_interfaces_text_errors():
    results = {
        "show interfaces statistics": CommandResult(
            command="show interfaces statistics",
            output=INTERFACES_STATS_TEXT,
        ),
    }
    metrics = extract_interface_metrics(results)
    by_name = {m.metric: m for m in metrics}

    assert by_name["iface_error_count"].value == 7.0
    assert by_name["iface_error_count"].is_counter is True


def test_interfaces_empty():
    assert extract_interface_metrics({}) == []


# ═══════════════════════════════════════════════════════════════════════
# System — XML path
# ═══════════════════════════════════════════════════════════════════════

def test_system_xml_storage():
    results = {
        "show system storage": CommandResult(
            command="show system storage", output="",
            structured=SYSTEM_STORAGE_XML,
        ),
    }
    metrics = extract_system_metrics(results)
    by_name = {m.metric: m for m in metrics}

    assert by_name["disk_used_pct"].value == 100.0


def test_system_xml_load_avg():
    results = {
        "show chassis routing-engine": CommandResult(
            command="show chassis routing-engine", output="",
            structured=ROUTING_ENGINE_XML,
        ),
    }
    metrics = extract_system_metrics(results)
    by_name = {m.metric: m for m in metrics}

    assert by_name["re_load_avg"].value == 0.25


# ═══════════════════════════════════════════════════════════════════════
# System — text fallback
# ═══════════════════════════════════════════════════════════════════════

def test_system_text_storage():
    results = {
        "show system storage": CommandResult(
            command="show system storage",
            output=SYSTEM_STORAGE_TEXT,
        ),
    }
    metrics = extract_system_metrics(results)
    by_name = {m.metric: m for m in metrics}

    assert by_name["disk_used_pct"].value == 100.0


def test_system_text_load_avg():
    results = {
        "show chassis routing-engine": CommandResult(
            command="show chassis routing-engine",
            output=ROUTING_ENGINE_TEXT,
        ),
    }
    metrics = extract_system_metrics(results)
    by_name = {m.metric: m for m in metrics}

    assert by_name["re_load_avg"].value == 0.25


def test_system_empty():
    assert extract_system_metrics({}) == []
