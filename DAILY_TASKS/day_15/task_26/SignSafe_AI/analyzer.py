import json
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def analyze_agreement(text):
    """
    Generates:
    - Summary
    - Risk Assessment
    - Clause Extraction
    """

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1
    )

    prompt = f"""
You are a legal agreement analysis assistant.

Analyze the agreement and return ONLY valid JSON.

Required format:

{{
    "summary": {{
        "purpose": "",
        "parties": "",
        "duration": "",
        "responsibilities": "",
        "conditions": ""
    }},
    "risk_assessment": {{
        "score": "Low | Medium | High",
        "risks": [
            {{
                "type": "",
                "reason": ""
            }}
        ]
    }},
    "clauses": [
        {{
            "name": "",
            "content": ""
        }}
    ]
}}

Important:

Identify risks such as:

- Automatic Renewal
- Early Termination Fee
- Data Sharing
- Non-Compete
- Excessive Liability
- Long Notice Period
- Confidentiality Restrictions

Extract important clauses such as:

- Payment Terms
- Termination Clause
- Renewal Clause
- Confidentiality Clause
- Liability Clause
- Penalty Clause

Agreement:

{text[:100000]}
"""

    response = llm.invoke(prompt)

    content = response.content.strip()

    if content.startswith("```json"):
        content = content.replace("```json", "")
        content = content.replace("```", "")
    elif content.startswith("```"):
        content = content.replace("```", "")

    try:
        return json.loads(content)

    except Exception:

        return {
            "summary": {
                "purpose": "Unable to generate summary",
                "parties": "-",
                "duration": "-",
                "responsibilities": "-",
                "conditions": "-"
            },
            "risk_assessment": {
                "score": "Medium",
                "risks": [
                    {
                        "type": "Processing Error",
                        "reason": "Could not parse AI response"
                    }
                ]
            },
            "clauses": []
        }