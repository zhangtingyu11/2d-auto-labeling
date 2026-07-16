# Data Security

Do not commit or upload any of the following:

- Company camera images or point clouds
- Human annotation exports containing private paths or metadata
- Model weights and TensorRT/ONNX artifacts
- Label Studio databases
- API tokens, passwords, cookies, or `.env` files

Use local absolute paths through environment variables. Before every push,
review the GitHub Desktop **Changes** list and confirm no data files appear.

If sensitive data is committed, stop pushing immediately. Removing a file in a
later commit does not remove it from Git history; rotate exposed credentials and
rewrite history before publishing again.
