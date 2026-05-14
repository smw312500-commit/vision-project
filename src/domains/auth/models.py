from flask_security.models import fsqla_v3

from src.main import db


# roles_users = db.Table(
#     "roles_users",
#     db.metadata,
#     Column("user_id", ForeignKey("user.id"), primary_key=True),
#     Column("role_id", ForeignKey("role.id"), primary_key=True),
# )


class Role(db.Model, fsqla_v3.FsRoleMixin):
    pass
    # id: Mapped[int] = mapped_column(primary_key=True)
    # name: Mapped[str] = mapped_column(unique=True, nullable=False)
    # description: Mapped[str] = mapped_column()
    #
    # users: Mapped[List["User"]] = relationship(
    #     secondary=roles_users, back_populates="roles"
    # )
