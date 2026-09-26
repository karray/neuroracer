FROM node:24-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci --ignore-scripts
# The npm release omits its custom loaders; fetch the matching release source.
ADD --checksum=sha256:838d3e1e6587efa761c978449e88a353e883ac475f58c9a600acf2270ce0635f https://codeload.github.com/gazebo-web/gzweb/tar.gz/519d4cf852237217908175c13154434503e2e658 /tmp/gzweb.tar.gz
RUN tar -xzf /tmp/gzweb.tar.gz -C /tmp && cp -r /tmp/gzweb-519d4cf852237217908175c13154434503e2e658/include node_modules/gzweb/
COPY app.js patch-gzweb.mjs ./
RUN npm run build

FROM nginx:1.28-alpine
COPY nginx.conf /etc/nginx/nginx.conf
COPY index.html /usr/share/nginx/html/index.html
COPY --from=build /app/dist /usr/share/nginx/html
