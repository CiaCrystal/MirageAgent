from marshmallow import Schema, ValidationError, fields, post_load, pre_load, validates


class OwnerSchema(Schema):
    name = fields.String(required=True, description="Owner name")
    email = fields.Email(missing=None)

    class Meta:
        ordered = True


class PetSchema(Schema):
    name = fields.String(required=True)
    species = fields.String(default="cat", missing="cat")
    age = fields.Integer(validate=lambda v: v >= 0)
    weight = fields.Number()
    owner = fields.Nested(OwnerSchema, missing=None)
    friends = fields.List(fields.Nested("self", exclude=("friends",)), missing=list)

    @validates("name")
    def check_name(self, value):
        if not value.strip():
            raise ValidationError("Name must not be blank.")

    @pre_load(pass_many=True)
    def strip_names(self, data, many, **kwargs):
        items = data if many else [data]
        for item in items:
            if isinstance(item.get("name"), str):
                item["name"] = item["name"].strip()
        return data

    @post_load
    def tag(self, data, **kwargs):
        data["tagged"] = True
        return data


class GreetingSchema(Schema):
    greeting = fields.Function(lambda obj, context: f"{context['prefix']} {obj['name']}")


def greet(pet, prefix="Hi"):
    return GreetingSchema(context={"prefix": prefix}).dump(pet)
