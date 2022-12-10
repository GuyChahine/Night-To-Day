from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config['SECRET KEY'] = "heozhirgurnbviueruhfg"
    
    from .views import views
    
    app.register_blueprint(views, url_predix='/')
    
    return app