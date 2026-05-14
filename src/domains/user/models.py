from typing import List

from flask_security.models import fsqla_v3
from sqlalchemy.orm import Mapped, relationship

from src.main import login_manager, db


class User(db.Model, fsqla_v3.FsUserMixin):
    images: Mapped[List["UserImage"]] = relationship(back_populates="user")
    videos: Mapped[List["UserVideo"]] = relationship(back_populates="user")
    # id: Mapped[int] = mapped_column(primary_key=True)
    # # username: Mapped[str] = mapped_column(index=True)
    # email: Mapped[str] = mapped_column(unique=True, nullable=False)
    # # password_hash: Mapped[str] = mapped_column(nullable=False)
    # password: Mapped[str] = mapped_column()
    # active: Mapped[bool] = mapped_column(nullable=False)
    # fs_uniquifier: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    #
    # roles: Mapped[List["Role"]] = relationship(
    #     secondary=roles_users, back_populates="users"
    # )

    # @property
    # def password(self):
    #     raise AttributeError("password is not a readable attribute")
    #
    # @password.setter
    # def password(self, password):
    #     self.password_hash = generate_password_hash(password)
    #
    # def verify_password(self, password):
    #     return check_password_hash(self.password_hash, password)

    # def is_duplicate_email(self):
    #     return User.query.filter_by(email=self.email).first() is not None


@login_manager.user_loader
def load_user(user_id):
    return User.query.filter_by(fs_uniquifier=user_id).first()
