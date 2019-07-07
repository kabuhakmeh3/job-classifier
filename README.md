# job-classifier

**Note:** ML/NLP model development and training was performed locally. Once
additional data is labeled, the model will be re-trained. There are several
demo notebooks available in this repository. One explores the types of jobs
available in a daily feed. The others demonstrate and compare the use of several techniques to classify text data. These are demonstrations only and should not be used in production. The current production model is a logistic regression binary classifier using a bag of words representation of text data. This is in the process of being updated to a multinomial naive Bayes classifier to handle multiple labels. Future development will consider complement naive Bayes to handle imbalanced class representations in training data.


### Development of NLP models to classify job postings

**Motivation**

Job boards have millions of postings. An individual xml file can be 15+ GB.
Job search sites are paid by link clicks.
Postings for skilled positions receive low conversion rates in when shown to an
advertising audience from social media platforms.
Featuring less-skilled positions results in lower ad-spend and higher engagement.
Identifying whether a job is low or high skilled is desirable.
Specific focus given to gig-economy jobs.

**Next Steps**

Model testing notebook demo

Full model testing & re-training

Automated updating embedded ad-copy links

Update workflow to use hdfs or spark instead of memory intensive ec2 instances

### Documentation

**Deploying on AWS**

Ensure full (not relative) paths to the following are provided

+ **keyfile** (.keys directory) - include bucket, url, target name, etc
+ **model** (models directory) - path

Only the classify_jobs.py script needs to be executed to generate an output.
Run times can be scheduled using a crontab. If a larger ec2 is required, use
a smaller instance to start and stop the larger instance where the job will run.

Sample cron commands (ensure you are calling the correct python installation)

Place in crontab of smaller (handler) instance

```
15 * * * cd /path/to/aws_tools && /usr/bin/python3 ./start_instance.py

50 15 * * * cd /path/to/aws_tools && /usr/bin/python3 ./stop_instance.py
```

Place in crontab of production instance

```
15 15 * * * cd /path/to/scripts && /usr/bin/python3 ./classify_partners.py
```

To create daily job reports

```
0 16 * * * cd /path/to/scripts && /usr/bin/python3 ./job_report.py
```
