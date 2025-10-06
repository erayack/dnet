
.PHONY: mdns #         | Show dns-sd services
mdns:
		dns-sd -Q _dnet_p2p._tcp.local. PTR

.PHONY: lint #         | Run linter
lint:
	  uvx ruff check

.PHONY: format #       | Format code
format:
		uvx ruff format

.PHONY: protos #       | Generate protobuf files
protos:
		uv run ./srcipts/generate_protos.py

.PHONY: help #         | List targets
help:                                                                                                                    
		@grep '^.PHONY: .* #' Makefile | sed 's/\.PHONY: \(.*\) # \(.*\)/\1 \2/' | expand -t20