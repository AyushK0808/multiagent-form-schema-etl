def load_schema():
    return {
        "form_name": "NDA_Form",
        "fields": {
            "effective_date": {
                "type": "date",
                "section": "Effective"
            },
            "termination_notice": {
                "type": "string",
                "section": "Termination"
            },
            "governing_law": {
                "type": "string",
                "section": "Governing"
            }
        }
    }
