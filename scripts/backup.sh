#!/bin/bash
# Quantum Planner Backup Script

set -e

BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "Starting backup at $(date)"

# Database backup
echo "Backing up PostgreSQL database..."
pg_dump -h $POSTGRES_HOST -U $POSTGRES_USER -d quantum_planner | gzip > "$BACKUP_DIR/database.sql.gz"

# Redis backup
echo "Backing up Redis data..."
redis-cli --rdb "$BACKUP_DIR/redis.rdb"

# Application configuration backup
echo "Backing up configuration files..."
tar -czf "$BACKUP_DIR/config.tar.gz" /app/config/

# Upload to cloud storage (AWS S3)
if [ "$AWS_S3_BUCKET" ]; then
    echo "Uploading backup to S3..."
    aws s3 cp "$BACKUP_DIR" "s3://$AWS_S3_BUCKET/backups/$(basename $BACKUP_DIR)" --recursive
fi

# Cleanup old local backups (keep 7 days)
find /backups -name "20*" -type d -mtime +7 -exec rm -rf {} +

echo "Backup completed at $(date)"
