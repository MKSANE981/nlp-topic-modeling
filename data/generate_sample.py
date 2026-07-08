"""Generate a sample dataset of tech posts for demo purposes."""
import pandas as pd, random

random.seed(42)

ML_POSTS = [
    "How to fine-tune a BERT model for text classification using HuggingFace transformers library with LoRA adapters",
    "Training neural network with gradient descent optimizer and learning rate scheduler for better convergence",
    "Implementing random forest classifier with scikit-learn cross validation and hyperparameter tuning grid search",
    "Deep learning convolutional neural network image classification with PyTorch transfer learning pretrained weights",
    "Natural language processing sentiment analysis using transformer embeddings and attention mechanism",
    "Gradient boosting XGBoost model feature importance SHAP values interpretability machine learning pipeline",
    "Regularization techniques dropout batch normalization overfitting prevention deep neural network training",
    "Clustering algorithm k-means unsupervised learning dimensionality reduction PCA embeddings visualization",
    "Recurrent neural network LSTM sequence modeling time series forecasting training backpropagation",
    "Model evaluation metrics precision recall F1 score ROC AUC confusion matrix cross validation",
    "Transformer attention mechanism self-attention multi-head bert gpt language model pretraining",
    "Reinforcement learning reward policy gradient agent environment training deep Q-network",
    "Support vector machine kernel trick classification regression SVM hyperplane margin optimization",
    "Feature engineering preprocessing normalization encoding categorical variables missing data imputation",
    "Ensemble methods bagging boosting stacking model combination prediction accuracy improvement",
    "Generative adversarial network GAN image synthesis discriminator generator training loss",
    "Word embeddings word2vec fasttext semantic similarity cosine distance vector representation",
    "Bayesian optimization hyperparameter search neural architecture probabilistic model acquisition function",
    "Data augmentation techniques training regularization image transformations text paraphrasing synthetic data",
    "Multi-task learning shared representation domain adaptation transfer learning fine-tuning pretrained models",
]

WEB_POSTS = [
    "How to build REST API with FastAPI Python asynchronous endpoints request validation response models",
    "React hooks useState useEffect component lifecycle state management performance optimization rendering",
    "Docker containerization deployment microservices orchestration Kubernetes pods services ingress configuration",
    "JavaScript async await promises fetch API error handling HTTP requests frontend backend communication",
    "CSS flexbox grid layout responsive design mobile first breakpoints media queries styling components",
    "Authentication JWT tokens OAuth2 session management middleware security headers CORS configuration",
    "GraphQL schema resolvers mutations queries subscription real-time data client Apollo server",
    "WebSocket real-time communication bidirectional server push notification event driven architecture",
    "TypeScript interfaces generics type safety compiler configuration strict mode frontend development",
    "Vue React Angular framework comparison component state routing lifecycle methods SPA frontend",
    "Next.js server-side rendering static generation hydration performance SEO optimization deployment",
    "PostgreSQL database schema design indexes foreign keys transactions ACID properties ORM SQLAlchemy",
    "Redis caching session storage pub-sub message broker queue background jobs Celery worker",
    "Nginx reverse proxy load balancing SSL certificate static files configuration upstream server",
    "CI/CD pipeline GitHub Actions automated testing deployment workflow build artifact container registry",
    "Webpack bundling code splitting lazy loading optimization tree shaking module federation frontend",
    "OAuth Google authentication social login callback endpoint token refresh user profile API",
    "Unit testing pytest mock fixtures integration testing coverage report test driven development",
    "API rate limiting throttling pagination cursor offset authentication middleware express router",
    "Microservices architecture service discovery message queue RabbitMQ event sourcing CQRS pattern",
]

DB_POSTS = [
    "SQL query optimization index usage explain plan join performance database tuning",
    "NoSQL MongoDB document store aggregation pipeline map reduce indexing sharding replication",
    "Data warehouse ETL pipeline transformation loading Spark distributed processing large scale",
    "Database normalization third normal form foreign key constraint referential integrity schema design",
    "Time series database InfluxDB Prometheus metrics collection aggregation retention policy query",
    "Elasticsearch full-text search inverted index analyzer tokenizer mapping relevance scoring",
    "Data lake architecture parquet format columnar storage partitioning Hive metastore query engine",
    "MySQL replication master slave failover clustering high availability backup restore recovery",
    "Redis sorted sets hash maps data structures caching expiration eviction policy performance",
    "Cassandra distributed database eventual consistency replication factor partition key wide column",
    "Spark DataFrame operations lazy evaluation transformation action RDD partition optimization",
    "Airflow DAG task dependency scheduling ETL orchestration pipeline monitoring retry failure",
    "dbt data build tool SQL transformation testing documentation lineage version control analytics",
    "Snowflake cloud data warehouse virtual warehouse query optimization materialized views clustering",
    "BigQuery SQL analytics GCP partitioned tables streaming insert scheduled query cost optimization",
    "Data modeling star schema dimension fact table OLAP OLTP reporting analytics business intelligence",
    "Database migration Alembic schema version control rollback up down revision history",
    "PostgreSQL window functions CTE recursive query lateral join analytical processing",
    "Kafka message broker topic partition consumer group offset producer streaming data pipeline",
    "Delta Lake ACID transactions schema enforcement time travel data versioning lakehouse architecture",
]

DEVOPS_POSTS = [
    "Terraform infrastructure as code AWS EC2 VPC security group provisioning state backend",
    "Kubernetes deployment scaling horizontal pod autoscaler resource limits requests liveness readiness",
    "Ansible playbook role task inventory automation configuration management idempotent execution",
    "Prometheus alerting rules Grafana dashboard metrics scraping service monitor visualization",
    "Git branching strategy merge rebase conflict resolution pull request code review workflow",
    "Linux system administration bash scripting cron job process monitoring log rotation systemd",
    "CloudFormation stack template resource output parameter condition mapping deployment update",
    "Helm chart values template deployment service ingress configmap secret Kubernetes package",
    "Packer image builder AMI Docker base image provisioner artifact pipeline automation",
    "Network firewall rules iptables security group VPC peering subnet routing table NAT gateway",
    "SSL TLS certificate Let's Encrypt ACME challenge renewal automation certbot nginx configuration",
    "Logging ELK stack Elasticsearch Logstash Kibana log aggregation parsing filter visualization",
    "Blue green deployment canary release traffic shifting feature flag rollback strategy zero downtime",
    "AWS Lambda serverless function trigger event API Gateway S3 DynamoDB IAM role permission",
    "Vault secrets management dynamic credentials database PKI certificate AWS IAM policy rotation",
    "Monitoring SLA SLO error budget alerting on-call incident postmortem reliability engineering",
    "Container security scanning vulnerability CVE image registry policy admission webhook OPA",
    "Jenkins pipeline groovy declarative scripted stage parallel agent artifact build test deploy",
    "Chaos engineering failure injection resilience testing latency fault tolerance service mesh Istio",
    "Cost optimization reserved instances spot fleet autoscaling rightsizing cloud resource tagging",
]

rows = []
for posts, topic in [(ML_POSTS, "ml_ai"), (WEB_POSTS, "web_dev"), (DB_POSTS, "databases"), (DEVOPS_POSTS, "devops")]:
    for post in posts:
        rows.append({"body": post, "true_topic": topic})

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("data/posts.csv", index=False)
print(f"Saved {len(df)} posts to data/posts.csv")