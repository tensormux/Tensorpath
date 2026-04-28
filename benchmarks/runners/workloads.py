"""Test prompts for benchmarking. Varied lengths to get a realistic distribution."""

# short prompts — typical chat turns (~30-60 input tokens)
CHAT_SHORT = [
    "Explain what a load balancer does in 2-3 sentences.",
    "What's the difference between TCP and UDP?",
    "Write a Python function that checks if a string is a palindrome.",
    "What are the pros and cons of microservices vs monoliths?",
    "Explain the CAP theorem simply.",
    "How does DNS resolution work?",
    "What is a race condition and how do you prevent it?",
    "Describe the difference between authentication and authorization.",
]

# medium prompts — more context (~100-200 input tokens)
CHAT_MEDIUM = [
    (
        "I'm building a REST API for an e-commerce platform. We have products, "
        "orders, and users. The API needs to handle about 500 requests per second "
        "at peak. We're using PostgreSQL as the database and Redis for caching. "
        "What are the key things I should think about for the API design? "
        "Focus on pagination, error handling, and rate limiting."
    ),
    (
        "Our team is debating whether to use Kubernetes or just plain Docker Compose "
        "for our deployment. We have about 15 microservices, 3 environments (dev, staging, prod), "
        "and a team of 5 backend engineers. Our current infra is on AWS. "
        "We're spending about $2000/month on EC2. "
        "What would you recommend and why?"
    ),
    (
        "I have a Python application that processes CSV files. Each file is about 500MB "
        "with 10 million rows. The processing involves parsing dates, computing rolling "
        "averages over a 30-day window, and joining with a reference table of 50K rows. "
        "Right now it takes about 45 minutes per file using pandas. "
        "How can I make this faster? Consider both algorithmic and infrastructure approaches."
    ),
    (
        "We're designing a notification system that needs to handle email, SMS, push notifications, "
        "and in-app notifications. Users should be able to set preferences per channel and per "
        "notification type. We expect about 1 million notifications per day. "
        "Walk me through the architecture you'd use, including message queues, "
        "delivery tracking, and retry logic."
    ),
]

# long prompts — summarization style (~300-500 input tokens)
SUMMARIZATION = [
    (
        "Summarize the key points from this technical discussion:\n\n"
        "The debate around monolithic vs microservice architectures has evolved significantly "
        "over the past decade. Initially, microservices were seen as the solution to all "
        "scalability problems. Companies like Netflix, Amazon, and Uber famously adopted "
        "microservices and published extensively about their benefits. The core arguments "
        "were clear: independent deployment, technology diversity, team autonomy, and "
        "fine-grained scaling. However, the industry has since learned that microservices "
        "come with substantial costs. Distributed tracing, service mesh management, "
        "network latency between services, data consistency across boundaries, and the "
        "operational burden of managing dozens or hundreds of services have caused many "
        "teams to reconsider. Some notable companies have even moved back toward more "
        "monolithic designs — or at least toward larger services with clearer boundaries. "
        "The term 'modular monolith' has gained popularity as a middle ground: a single "
        "deployable unit with strong internal module boundaries that can be split later "
        "if needed. The key insight is that the right architecture depends heavily on "
        "team size, organizational structure, deployment frequency, and the actual "
        "scalability requirements of the system. A startup with 5 engineers almost certainly "
        "doesn't need microservices. A company with 500 engineers working on the same "
        "product almost certainly does. The architecture should match the organization, "
        "not the other way around."
    ),
]


def get_prompts(workload: str = "chat", count: int = 20) -> list[str]:
    """Get a mix of prompts for benchmarking."""
    if workload == "chat":
        pool = CHAT_SHORT * 3 + CHAT_MEDIUM * 2
    elif workload == "summarization":
        pool = SUMMARIZATION * 5 + CHAT_MEDIUM * 3
    elif workload == "codegen":
        pool = CHAT_SHORT * 4  # code questions are usually short
    else:
        pool = CHAT_SHORT * 2 + CHAT_MEDIUM * 2

    # repeat to fill count
    result = []
    while len(result) < count:
        result.extend(pool)
    return result[:count]
