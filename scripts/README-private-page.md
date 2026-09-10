# Publishing the private project page

Keep the editable page source outside the published branch. The protected
GitHub Pages branch contains only a password gate and encrypted binary assets.
The PDF and videos are decrypted in the browser after successful login; the
other assets are embedded inside the encrypted page.

`build_private_page.cjs` uses Node.js built-ins and accepts the existing
StatiCrypt password on standard input. It verifies that password against the
previous protected page before building anything, retaining the existing salt
and remember-me behavior. Never pass a password as a command-line argument or
commit a password/configuration file.

```bash
# SOURCE contains the editable index.html, scripts, styles, and assets.
# GATE is a local copy of the previous protected index.html.
# OUTPUT must be a new or empty directory.
read -r -s BVB_PAGE_PASSWORD
printf '%s' "$BVB_PAGE_PASSWORD" | node scripts/build_private_page.cjs \
  --source "$SOURCE" --gate "$GATE" --output "$OUTPUT"
unset BVB_PAGE_PASSWORD
```

The build verifies the prior password, embeds the page's scripts and small
assets, and encrypts the PDF and videos separately for on-demand loading.
Before publishing, verify both failed and successful login, remember-me,
video playback, and downloads. Inspect the exact commit tree for plaintext
assets, local paths, secrets, and private authoring notes. Push only the
validated protected output, based on the current remote Pages branch.
