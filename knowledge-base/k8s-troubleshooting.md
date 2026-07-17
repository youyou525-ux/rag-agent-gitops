# Kubernetes Troubleshooting Guide

## Pod Pending

If a Pod is in Pending state, use kubectl describe pod <pod-name> to check events. Common reasons are insufficient resources or unbound PVCs.

## CrashLoopBackOff

Check container logs with kubectl logs <pod-name>. Common causes include application errors, missing config, or wrong command.

## Service Not Reachable

Verify endpoints with kubectl get endpoints <service-name>. If empty, check label selectors match between Service and Pods.
