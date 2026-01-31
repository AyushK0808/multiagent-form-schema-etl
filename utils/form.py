class FormInstance:
    def __init__(self, schema):
        self.form_name = schema["form_name"]
        self.fields = {k: None for k in schema["fields"]}

    def fill(self, field, value):
        self.fields[field] = value

    def to_json(self):
        return {
            "form": self.form_name,
            "fields": self.fields
        }
