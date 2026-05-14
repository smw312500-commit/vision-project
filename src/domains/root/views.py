from flask import Blueprint, render_template
from flask_security import roles_required

root_views = Blueprint(
    "root",
    __name__,
    template_folder="templates",
    static_folder="static",
)


@root_views.route("/")
def index():
    return render_template("root/index.html")


@root_views.route("/admin")
@roles_required("admin")
def about():
    return render_template("root/admin.html")
