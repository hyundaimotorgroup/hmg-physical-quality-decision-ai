# Web app

`pip install -e ".[web]"` then `pqdai serve --domain weld` (or `webapp/start_linux.sh`, `start_mac.command`, `start_windows.bat`) and open http://127.0.0.1:8000.

* Upload videos (queued, one at a time) or images (synchronous); progress stepper, live frame preview, result overlay and dedup-audit report, per-day history.
* Domain via `PQDAI_DOMAIN`; weights via `PQDAI_DETECTOR_WEIGHTS` / `PQDAI_CLASSIFIER_WEIGHTS`.
* **Security**: the login is a name picker without a password (`webapp/backend/data/users.example.json`; copy to `users.json` and edit). It records who analysed what; it is **not access control**. Uploads are unrestricted. Run on trusted networks only or behind an authenticating reverse proxy.
* `webapp/uploads`, `webapp/results` and `history.json` grow without cleanup.
* The bundled front-end texts are Korean and weld-oriented (first domain); the API is domain-neutral (`objects`, `overall_class`).
