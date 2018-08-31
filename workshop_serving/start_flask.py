#!/usr/bin/env python

from flask import Flask

app = Flask(__name__)


@app.route("/")
def hello():
  return "Hello World!"


def main():
  host = "0.0.0.0"
  port = 8500
  app.run(host=host, port=port)


if __name__ == "__main__":
  main()
