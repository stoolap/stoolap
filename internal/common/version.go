/*
Copyright 2025 Stoolap Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package common

const (
	// VersionMajor is the major version of the driver
	VersionMajor = "0"
	// VersionMinor is the minor version of the driver
	VersionMinor = "1"
	// VersionPatch is the patch version of the driver
	VersionPatch = "2"
	// VersionSuffix is the suffix of the driver version
	VersionSuffix = "40ab0092" // git commit hash

	// VersionString is the version string of the driver
	VersionString = "Stoolap v" + VersionMajor + "." + VersionMinor + "." + VersionPatch + "-" + VersionSuffix
)
