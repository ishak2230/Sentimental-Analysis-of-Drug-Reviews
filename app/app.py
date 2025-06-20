from flask import Flask,render_template,request,jsonify from flask_mysqldb import MySQL
app = Flask( name ) mysql = MySQL(app)

#database connectivity information

app.config["MYSQL_HOST"] = "localhost" app.config["MYSQL_USER"] = "root" app.config["MYSQL_PASSWORD"] = ""
 
app.config["MYSQL_DB"] = "csv_db" app.config["MYSQL_CURSORCLASS"] = "DictCursor"

@app.route("/") def index():
return render_template("index.html") @app.route("/livesearch",methods=["POST","GET"]) def livesearch():
searchbox = request.form.get("text") cursor = mysql.connection.cursor()
query = "SELECT r.drugName,r.total_pred_mean,s.information FROM recommendation r left outer join scraping s on r.drugName=s.drugName where r.Conditions LIKE '{}%' order by r.Conditions,r.total_pred_mean desc limit 5".format(searchbox) cursor.execute(query)
result = cursor.fetchall() return jsonify(result)

if 	name 	== " main ": app.run(debug=True)
