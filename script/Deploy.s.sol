// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Script, console2} from "forge-std/Script.sol";
import {OnChainLLM} from "../src/OnChainLLM.sol";

contract Deploy is Script {
    function run() external returns (OnChainLLM) {
        vm.startBroadcast();
        OnChainLLM llm = new OnChainLLM();
        vm.stopBroadcast();
        console2.log("OnChainLLM deployed at:", address(llm));
        return llm;
    }
}
