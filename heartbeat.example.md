# JACE Heartbeat Checklist
# Each line is a monitoring instruction executed on every heartbeat cycle.
# Edit this file or use natural language: "add a heartbeat check for ..."

- Alert if any device has unresolved critical findings older than 30 minutes
- Verify all BGP peers are in Established state on every device
- Check that no device has RE CPU usage above 80%
- If any interface error counter delta exceeds 1000/hour, investigate the cause
