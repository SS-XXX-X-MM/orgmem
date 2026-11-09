date: 2026-04-14

# Incident review: duplicated invoices (INC-212)

Root cause: a retry in the webhook handler ran twice without an idempotency key.

Action items agreed by the team:
- Every payment webhook handler must check an idempotency key before writing.
- Hotfixes to billing need one approval from the billing owner and a green CI run.

Note: Priya is on leave in May; Dana covers billing questions during that time.
