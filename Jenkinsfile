pipeline {
    agent any
    environment {
        DOCKERHUB_CREDENTIALS_ID = 'docker-hub-credentials'
    }
    stages {
        stage('Build Docker Image') {
            steps {
                script {
                    docker.build('myapp:${BUILD_NUMBER}')
                }
            }
        }
        stage('Push to Docker Hub') {
            steps {
                script {
                    docker.withRegistry('https://index.docker.io/v1/', DOCKERHUB_CREDENTIALS_ID) {
                        docker.image('myapp:${BUILD_NUMBER}').push()
                    }
                }
            }
        }
    }
    post {
        success {
            mail(to: 'i200968@nu.edu.pk, i200425@nu.edu.pk',
                 subject: "Jenkins Job Success: ${env.JOB_NAME} [${env.BUILD_NUMBER}]",
                 body: "Jenkins job successfully completed. Docker image has been pushed.")
        }
    }
}
