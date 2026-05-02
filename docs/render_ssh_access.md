# Render SSH 备忘

## 服务

- Render web service: `keiba-web`
- Region host: `ssh.oregon.render.com`
- SSH target: `srv-d6qm9j9aae7s739j7kkg@ssh.oregon.render.com`

## 常用命令

只读连通性检查：

```bash
ssh -o BatchMode=yes -o UpdateHostKeys=no srv-d6qm9j9aae7s739j7kkg@ssh.oregon.render.com "pwd && hostname"
```

查看项目根目录：

```bash
ssh -o BatchMode=yes -o UpdateHostKeys=no srv-d6qm9j9aae7s739j7kkg@ssh.oregon.render.com "ls -la /opt/render/project/src"
```

查看线上数据目录：

```bash
ssh -o BatchMode=yes -o UpdateHostKeys=no srv-d6qm9j9aae7s739j7kkg@ssh.oregon.render.com "ls -la /opt/render/project/src/pipeline/data"
```

## 已确认的线上路径

- 项目根目录：`/opt/render/project/src`
- pipeline 目录：`/opt/render/project/src/pipeline`
- 持久化数据目录：`/opt/render/project/src/pipeline/data`

## 备注

- 本机已能连接该 Render 实例。
- 这台机器当前没有保存 Render CLI 配置，后续直接用上面的 SSH 目标即可。
- 连接时若出现 `client_global_hostkeys_prove_confirm: server gave bad signature for ED25519 key 0`，先加 `-o UpdateHostKeys=no` 规避，再继续做只读排查。
- Render cron 服务不支持 SSH；这里记录的是 `keiba-web` 这台 web service。
