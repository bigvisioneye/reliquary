module.exports = {
    apps: [
      {
        name: "reliquary-miner",
        cwd: "/workspace/reliquary",
        script: "scripts/run_miner_pm2.sh",
        interpreter: "bash",
        autorestart: true,
        max_restarts: 20,
        restart_delay: 5000,
        exp_backoff_restart_delay: 10000,
        kill_timeout: 30000,
        time: true,
        out_file: "/workspace/reliquary/logs/miner.out.log",
        error_file: "/workspace/reliquary/logs/miner.err.log",
        merge_logs: true,
      },
    ],
  };