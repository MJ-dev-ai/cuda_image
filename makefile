ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
CONFIG := $(if $(filter debug,$(MAKECMDGOALS)),Debug,Release)
.DEFAULT_GOAL := help
.PHONY: help build run clean rebuild debug release
ifneq ($(words $(filter debug release,$(MAKECMDGOALS))),0)
ifneq ($(words $(filter debug release,$(MAKECMDGOALS))),1)
$(error Select debug OR release)
endif
endif
help:
	@echo make build/run/clean/rebuild [debug/release]
build run clean rebuild:
	powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(ROOT)build_tasks.ps1" -Action $@ -Configuration $(CONFIG) -RunArgs '$(ARGS)'
debug release:
	@echo Configuration: $(CONFIG)
