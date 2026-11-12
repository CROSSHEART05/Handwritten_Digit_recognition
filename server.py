import http.server
import socketserver
import webbrowser
import os

PORT = 8000

class QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {args[0]}")

if __name__ == "__main__":
    web_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(web_dir)

    print(f"==================================================")
    print(f"  MNIST Terminal Classifier Server Running")
    print(f"  URL: http://localhost:{PORT}")
    print(f"==================================================")

    webbrowser.open(f"http://localhost:{PORT}")

    with socketserver.TCPServer(("", PORT), QuietHTTPRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[*] Server stopped.")
