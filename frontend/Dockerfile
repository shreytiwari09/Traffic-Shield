# Multi-stage: build the React bundle with Node, then serve it from nginx.
#
# The runtime image carries no Node, no node_modules and no source — just static
# assets plus nginx. That is both much smaller and closer to how this would
# actually be deployed than shipping `vite dev` in a container.

# --- build stage ------------------------------------------------------------
FROM node:22-alpine AS build

WORKDIR /app

# `playwright` is a devDependency here (used by the _check*.mjs scratch scripts,
# not by the app). Its postinstall downloads several hundred MB of browsers,
# which this image will never run — skipping it saves most of the build time.
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1

# npm ci needs both files and installs exactly the lockfile — reproducible, and
# cached until the lockfile itself changes.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# --- runtime stage ----------------------------------------------------------
FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

# nginx's own image already sets an appropriate CMD; declaring it explicitly
# here would only risk drifting from upstream.
