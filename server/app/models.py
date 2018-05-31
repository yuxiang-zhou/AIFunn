from app import db

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), index=True, unique=True)
    data = db.Column(db.String(120), index=True, unique=True)

    def __repr__(self):
        return '<Image %r>' % (self.name)
