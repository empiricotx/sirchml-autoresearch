# Frontend Startup

Start the read-only session viewer with Docker:

```bash
docker compose up --build
```

When the containers are running, open:

- Frontend: `http://localhost:5173`
- API: `http://localhost:8000`

## Notes

- The API mounts `sessions/` read-only from this repository.
- The frontend reads data from the API, not directly from the filesystem.
- If you change frontend or API files, rerun `docker compose up --build` to rebuild the containers.
