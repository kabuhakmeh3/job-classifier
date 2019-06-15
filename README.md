# job-classifier

Exploring the use of NLP models to classify job postings

**Deploying on AWS**

Ensure full (not relative) paths to the following are provided

+ keyfile (.keys directory) - include bucket, url, target name
+ model (models directory) - path

Only the classify_jobs.py script needs to be executed to generate an output.
Run times can be scheduled using a crontab. If a larger ec2 is required, use
a smaller instance to start and stop the larger instance where the job will run.

Sample cron commands (ensure you are calling the correct python installation)

Place in crontab of smaller (handler) instance

```
17 * * * cd /path/to/aws_tools && /usr/bin/python3 ./start_instance.py

50 17 * * * cd /path/to/aws_tools && /usr/bin/python3 ./stop_instance.py
```

Place in crontab of production instance

```
15 17 * * * cd /path/to/scripts && /usr/bin/python3 ./cron_classify_jobs.py
```

**TO-DO**

- EDA notebook < [current]
- prep/label initial training data
- model selection
- data cleaning/processing/validation workflow
- final training
- model deployment

**Motivation**

Job boards have millions of postings.
Job search sites are paid by link clicks.
Postings for skilled positions receive low conversion rates in when shown to an
advertising audience from social media platforms.
Featuring less-skilled positions results in lower ad-spend and higher return.
Identifying whether a job is low or high skilled is desirable.
Specific focus given to gig-economy jobs.

**Objectives**

Exploratory notebook

Predictive model

Serving daily job postings to feature in ads

Automating/updating ad-copy links

Delivering recommendations in an effective manner to ad-ops teams

Integrating with existing job partner parsing pipeline

Update workflow to use hdfs or spark instead of memory intensive ec2 instances
