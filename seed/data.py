"""The demo portfolio: six customer accounts, seven deals, one sales rep.

Every date is an offset in days from the moment the seed runs (negative is the
past), so a fresh database always looks current -- a deal seeded as "quiet for
four weeks" is quiet for four weeks whenever it is loaded, rather than drifting
into a year-old graveyard.

The rep is ``settings.ae_display_name`` (Maya Chen); the product being sold is
SecureFlow, an audit-logging platform. The deals are shaped so the
deterministic risk rules have something real to find:

*   Halvorsen -- healthy, late-stage, fully multi-threaded.
*   Brightwater -- economic buyer known but never in a meeting; overdue task.
*   Kestrel PCI -- single-threaded and stalled in evaluation.
*   Meridian -- gone quiet after discovery.
*   Ostara -- brand new, nothing to say yet.
*   Lumen Ridge / Kestrel Fraud Ops -- closed won and closed lost.

Identity for "already seeded?" is the natural key, never the id: account name,
contact email, deal name within its account, meeting title within its deal,
task title within its deal, and attendee name within its meeting.
"""

AE = "Maya Chen"

ACCOUNTS = [
    # ------------------------------------------------------------------ 1
    {
        "name": "Halvorsen Freight Group",
        "industry": "Logistics",
        "website": "halvorsenfreight.com",
        "employee_band": "1,000-5,000",
        "hq_region": "Northern Europe",
        "contacts": [
            {"key": "ingrid", "first_name": "Ingrid", "last_name": "Solberg",
             "title": "Chief Information Security Officer",
             "email": "ingrid.solberg@halvorsenfreight.com", "phone": "+47 22 41 18 30"},
            {"key": "lars", "first_name": "Lars", "last_name": "Eriksen",
             "title": "Head of Platform Engineering",
             "email": "lars.eriksen@halvorsenfreight.com", "phone": "+47 22 41 18 52"},
            {"key": "sofie", "first_name": "Sofie", "last_name": "Brandt",
             "title": "VP Finance",
             "email": "sofie.brandt@halvorsenfreight.com", "phone": "+47 22 41 18 07"},
        ],
        "deals": [
            {
                "name": "SecureFlow Enterprise - EU Hub Rollout",
                "stage": "negotiation",
                "value": 240000,
                "currency": "EUR",
                "win_probability": 70,
                "expected_close_in_days": 21,
                "created_days_ago": 92,
                "stage_history": [
                    ("qualification", -92, "Deal created"),
                    ("discovery", -80, "Discovery call booked with the CISO"),
                    ("evaluation", -58, "Technical deep dive completed"),
                    ("security_and_legal", -37, "Security review started"),
                    ("negotiation", -12, "Security sign-off received; commercials open"),
                ],
                "stakeholders": [
                    {"contact": "ingrid", "buying_role": "champion", "influence": "high",
                     "sentiment": "positive", "is_primary": True,
                     "notes": "Sponsoring the project ahead of the Q1 regulator audit."},
                    {"contact": "lars", "buying_role": "technical", "influence": "medium",
                     "sentiment": "positive",
                     "notes": "Ran the load test; owns the ingestion pipeline."},
                    {"contact": "sofie", "buying_role": "economic_buyer", "influence": "high",
                     "sentiment": "neutral",
                     "notes": "Signs anything over EUR 200k. Wants a three-year price lock."},
                ],
                "meetings": [
                    {"title": "Discovery call", "meeting_type": "discovery",
                     "status": "completed", "days": -80, "minutes": 45,
                     "attendees": [AE, "Ingrid Solberg", "Lars Eriksen"]},
                    {"title": "Technical deep dive", "meeting_type": "technical_review",
                     "status": "completed", "days": -60, "minutes": 60,
                     "attendees": [AE, "Lars Eriksen"]},
                    {"title": "Security review", "meeting_type": "security_review",
                     "status": "completed", "days": -40, "minutes": 60,
                     "attendees": [AE, "Ingrid Solberg", "Lars Eriksen"]},
                    {"title": "Commercial negotiation", "meeting_type": "negotiation",
                     "status": "completed", "days": -9, "minutes": 40,
                     "attendees": [AE, "Ingrid Solberg", "Sofie Brandt"],
                     "transcript": "halvorsen-commercial-negotiation.txt"},
                    {"title": "Contract signature call", "meeting_type": "check_in",
                     "status": "scheduled", "days": 6, "minutes": 30},
                ],
                "tasks": [
                    {"title": "Send revised order form with three-year pricing",
                     "description": "Sofie asked for a 36-month term with a capped annual uplift.",
                     "due_in_days": 2, "priority": "high", "status": "open"},
                    {"title": "Confirm DPA redlines with Halvorsen legal",
                     "description": "Two open clauses: sub-processor notice period and audit rights.",
                     "due_in_days": -1, "priority": "urgent", "status": "open"},
                    {"title": "Share load-test results summary",
                     "due_in_days": -30, "priority": "medium", "status": "done"},
                ],
            },
        ],
    },
    # ------------------------------------------------------------------ 2
    {
        "name": "Brightwater Health Partners",
        "industry": "Healthcare",
        "website": "brightwaterhealth.org",
        "employee_band": "5,000+",
        "hq_region": "US East",
        "contacts": [
            {"key": "daniel", "first_name": "Daniel", "last_name": "Okafor",
             "title": "Director of Information Security",
             "email": "daniel.okafor@brightwaterhealth.org", "phone": "+1 617 555 0142"},
            {"key": "rachel", "first_name": "Rachel", "last_name": "Lindqvist",
             "title": "Chief Compliance Officer",
             "email": "rachel.lindqvist@brightwaterhealth.org", "phone": "+1 617 555 0118"},
            {"key": "kevin", "first_name": "Kevin", "last_name": "Marsh",
             "title": "Enterprise Architect",
             "email": "kevin.marsh@brightwaterhealth.org", "phone": "+1 617 555 0175"},
            {"key": "gregory", "first_name": "Gregory", "last_name": "Hale",
             "title": "Chief Financial Officer",
             "email": "gregory.hale@brightwaterhealth.org", "phone": "+1 617 555 0101"},
        ],
        "deals": [
            {
                "name": "Audit Log Retention - HIPAA Program",
                "stage": "security_and_legal",
                "value": 310000,
                "currency": "USD",
                "win_probability": 55,
                "expected_close_in_days": 38,
                "created_days_ago": 70,
                "stage_history": [
                    ("qualification", -70, "Deal created"),
                    ("discovery", -61, "Inbound from compliance team"),
                    ("evaluation", -45, "Architecture review scheduled"),
                    ("security_and_legal", -24, "Security questionnaire received"),
                ],
                "stakeholders": [
                    {"contact": "daniel", "buying_role": "champion", "influence": "high",
                     "sentiment": "positive", "is_primary": True,
                     "notes": "Owns the HIPAA audit-log remediation plan."},
                    {"contact": "rachel", "buying_role": "influencer", "influence": "high",
                     "sentiment": "neutral",
                     "notes": "Needs six-year retention to satisfy the compliance program."},
                    {"contact": "kevin", "buying_role": "technical", "influence": "medium",
                     "sentiment": "neutral",
                     "notes": "Concerned about Epic integration effort."},
                    {"contact": "gregory", "buying_role": "economic_buyer", "influence": "high",
                     "sentiment": "unknown",
                     "notes": "Final approver. Has not joined a call yet."},
                ],
                "meetings": [
                    {"title": "Discovery call", "meeting_type": "discovery",
                     "status": "completed", "days": -61, "minutes": 45,
                     "attendees": [AE, "Daniel Okafor", "Rachel Lindqvist"]},
                    {"title": "Architecture review", "meeting_type": "technical_review",
                     "status": "completed", "days": -45, "minutes": 60,
                     "attendees": [AE, "Daniel Okafor", "Kevin Marsh"]},
                    {"title": "Security questionnaire walkthrough", "meeting_type": "security_review",
                     "status": "completed", "days": -18, "minutes": 50,
                     "attendees": [AE, "Daniel Okafor", "Kevin Marsh", "Rachel Lindqvist"],
                     "transcript": "brightwater-security-walkthrough.txt"},
                    {"title": "Executive alignment with CFO", "meeting_type": "check_in",
                     "status": "scheduled", "days": 9, "minutes": 30},
                ],
                "tasks": [
                    {"title": "Return completed security questionnaire",
                     "description": "Sections 4 (encryption) and 7 (BAA) still outstanding.",
                     "due_in_days": -3, "priority": "high", "status": "open"},
                    {"title": "Draft BAA for Brightwater legal review",
                     "due_in_days": 5, "priority": "high", "status": "open"},
                    {"title": "Book executive sponsor meeting with the CFO",
                     "due_in_days": -6, "priority": "medium", "status": "done"},
                ],
            },
        ],
    },
    # ------------------------------------------------------------------ 3
    {
        "name": "Kestrel Payments",
        "industry": "Financial Services",
        "website": "kestrelpayments.co.uk",
        "employee_band": "500-1,000",
        "hq_region": "UK & Ireland",
        "contacts": [
            {"key": "oliver", "first_name": "Oliver", "last_name": "Grant",
             "title": "Head of Security Operations",
             "email": "oliver.grant@kestrelpayments.co.uk", "phone": "+44 20 7946 0321"},
            {"key": "fiona", "first_name": "Fiona", "last_name": "Byrne",
             "title": "Head of Fraud Operations",
             "email": "fiona.byrne@kestrelpayments.co.uk", "phone": "+44 20 7946 0388"},
        ],
        "deals": [
            {
                "name": "PCI DSS Logging Expansion",
                "stage": "evaluation",
                "value": 95000,
                "currency": "GBP",
                "win_probability": 35,
                "expected_close_in_days": 14,
                "created_days_ago": 85,
                "stage_history": [
                    ("qualification", -85, "Deal created"),
                    ("discovery", -78, "Discovery with SecOps"),
                    ("evaluation", -56, "Trial environment provisioned"),
                ],
                "stakeholders": [
                    {"contact": "oliver", "buying_role": "champion", "influence": "medium",
                     "sentiment": "positive", "is_primary": True,
                     "notes": "Only contact engaged so far."},
                ],
                "meetings": [
                    {"title": "Discovery call", "meeting_type": "discovery",
                     "status": "completed", "days": -78, "minutes": 30,
                     "attendees": [AE, "Oliver Grant"]},
                    {"title": "Product demo", "meeting_type": "demo",
                     "status": "completed", "days": -56, "minutes": 45,
                     "attendees": [AE, "Oliver Grant"]},
                    {"title": "Trial check-in", "meeting_type": "check_in",
                     "status": "completed", "days": -33, "minutes": 20,
                     "attendees": [AE, "Oliver Grant"]},
                ],
                "tasks": [
                    {"title": "Agree written evaluation success criteria with Oliver",
                     "due_in_days": -10, "priority": "high", "status": "open"},
                    {"title": "Identify a budget holder beyond SecOps",
                     "due_in_days": 4, "priority": "medium", "status": "open"},
                ],
            },
            {
                "name": "Fraud Ops Log Archive",
                "stage": "closed_lost",
                "value": 70000,
                "currency": "GBP",
                "win_probability": 0,
                "expected_close_in_days": -40,
                "closed_days_ago": 40,
                "created_days_ago": 120,
                "stage_history": [
                    ("qualification", -120, "Deal created"),
                    ("discovery", -110, None),
                    ("evaluation", -90, None),
                    ("closed_lost", -40, "Lost to the incumbent SIEM vendor on price"),
                ],
                "stakeholders": [
                    {"contact": "fiona", "buying_role": "economic_buyer", "influence": "high",
                     "sentiment": "negative", "is_primary": True,
                     "notes": "Chose to extend the existing SIEM contract."},
                    {"contact": "oliver", "buying_role": "influencer", "influence": "medium",
                     "sentiment": "positive"},
                ],
                "meetings": [
                    {"title": "Discovery call", "meeting_type": "discovery",
                     "status": "completed", "days": -110, "minutes": 30,
                     "attendees": [AE, "Fiona Byrne", "Oliver Grant"]},
                    {"title": "Pricing review", "meeting_type": "negotiation",
                     "status": "completed", "days": -48, "minutes": 30,
                     "attendees": [AE, "Fiona Byrne"]},
                ],
                "tasks": [],
            },
        ],
    },
    # ------------------------------------------------------------------ 4
    {
        "name": "Meridian Retail Co.",
        "industry": "Retail",
        "website": "meridianretail.com",
        "employee_band": "1,000-5,000",
        "hq_region": "US Midwest",
        "contacts": [
            {"key": "hannah", "first_name": "Hannah", "last_name": "Cole",
             "title": "IT Operations Manager",
             "email": "hannah.cole@meridianretail.com", "phone": "+1 312 555 0190"},
            {"key": "victor", "first_name": "Victor", "last_name": "Nguyen",
             "title": "Director of Infrastructure",
             "email": "victor.nguyen@meridianretail.com", "phone": "+1 312 555 0164"},
        ],
        "deals": [
            {
                "name": "Store Systems Audit Trail",
                "stage": "discovery",
                "value": 60000,
                "currency": "USD",
                "win_probability": 20,
                "expected_close_in_days": 60,
                "created_days_ago": 48,
                "stage_history": [
                    ("qualification", -48, "Deal created"),
                    ("discovery", -40, "Inbound demo request"),
                ],
                "stakeholders": [
                    {"contact": "hannah", "buying_role": "champion", "influence": "medium",
                     "sentiment": "neutral", "is_primary": True},
                    {"contact": "victor", "buying_role": "influencer", "influence": "medium",
                     "sentiment": "unknown"},
                ],
                "meetings": [
                    {"title": "Discovery call", "meeting_type": "discovery",
                     "status": "completed", "days": -29, "minutes": 35,
                     "attendees": [AE, "Hannah Cole", "Victor Nguyen"],
                     "transcript": "meridian-discovery.txt"},
                ],
                "tasks": [
                    {"title": "Send discovery recap and proposed next steps",
                     "due_in_days": -27, "priority": "medium", "status": "done"},
                    {"title": "Re-engage Hannah after four weeks without a reply",
                     "due_in_days": 1, "priority": "medium", "status": "open"},
                ],
            },
        ],
    },
    # ------------------------------------------------------------------ 5
    {
        "name": "Ostara Energy",
        "industry": "Energy & Utilities",
        "website": "ostaraenergy.com",
        "employee_band": "5,000+",
        "hq_region": "Nordics",
        "contacts": [
            {"key": "erik", "first_name": "Erik", "last_name": "Lindahl",
             "title": "OT Security Lead",
             "email": "erik.lindahl@ostaraenergy.com", "phone": "+46 8 555 210 44"},
        ],
        "deals": [
            {
                "name": "OT Network Monitoring Pilot",
                "stage": "qualification",
                "value": 45000,
                "currency": "EUR",
                "win_probability": 10,
                "expected_close_in_days": 90,
                "created_days_ago": 6,
                "stage_history": [
                    ("qualification", -6, "Deal created"),
                ],
                "stakeholders": [
                    {"contact": "erik", "buying_role": "influencer", "influence": "medium",
                     "sentiment": "unknown", "is_primary": True},
                ],
                "meetings": [
                    {"title": "Introductory call", "meeting_type": "discovery",
                     "status": "scheduled", "days": 3, "minutes": 30},
                ],
                "tasks": [
                    {"title": "Prepare pilot scoping questions for Erik",
                     "due_in_days": 2, "priority": "low", "status": "open"},
                ],
            },
        ],
    },
    # ------------------------------------------------------------------ 6
    {
        "name": "Lumen Ridge Software",
        "industry": "Software",
        "website": "lumenridge.io",
        "employee_band": "200-500",
        "hq_region": "US West",
        "contacts": [
            {"key": "nina", "first_name": "Nina", "last_name": "Patel",
             "title": "VP Engineering",
             "email": "nina.patel@lumenridge.io", "phone": "+1 415 555 0127"},
            {"key": "sam", "first_name": "Sam", "last_name": "Whitaker",
             "title": "Staff Site Reliability Engineer",
             "email": "sam.whitaker@lumenridge.io", "phone": "+1 415 555 0183"},
        ],
        "deals": [
            {
                "name": "SecureFlow Growth Plan",
                "stage": "closed_won",
                "value": 38000,
                "currency": "USD",
                "win_probability": 100,
                "expected_close_in_days": -12,
                "closed_days_ago": 12,
                "created_days_ago": 75,
                "stage_history": [
                    ("qualification", -75, "Deal created"),
                    ("discovery", -66, None),
                    ("evaluation", -50, "Two-week trial started"),
                    ("negotiation", -25, None),
                    ("closed_won", -12, "Signed annual Growth plan"),
                ],
                "stakeholders": [
                    {"contact": "nina", "buying_role": "economic_buyer", "influence": "high",
                     "sentiment": "positive", "is_primary": True},
                    {"contact": "sam", "buying_role": "champion", "influence": "medium",
                     "sentiment": "positive"},
                ],
                "meetings": [
                    {"title": "Discovery call", "meeting_type": "discovery",
                     "status": "completed", "days": -66, "minutes": 30,
                     "attendees": [AE, "Sam Whitaker"]},
                    {"title": "Trial kickoff", "meeting_type": "demo",
                     "status": "completed", "days": -50, "minutes": 45,
                     "attendees": [AE, "Sam Whitaker", "Nina Patel"]},
                    {"title": "Commercial review", "meeting_type": "negotiation",
                     "status": "completed", "days": -20, "minutes": 30,
                     "attendees": [AE, "Nina Patel"]},
                ],
                "tasks": [
                    {"title": "Kick off onboarding with the Lumen Ridge SRE team",
                     "due_in_days": -5, "priority": "high", "status": "done"},
                    {"title": "Schedule 30-day adoption review",
                     "due_in_days": 18, "priority": "medium", "status": "open"},
                ],
            },
        ],
    },
]
