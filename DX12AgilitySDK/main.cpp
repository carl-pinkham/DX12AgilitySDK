// d3d12.lib, dxgi.lib, and d3dcompiler.lib set in Project settings

//***************************************************************************************
// adapted from CameraAndDynamicIndexingApp.cpp by Frank Luna (C) 2015 All Rights Reserved.
//***************************************************************************************

// TO DO:

// change from raw pointers to ComPtr
// check HRESULTs
// validate m_hWnd
// debug
// exceptions
// reorganize code into separate header and implementation files

#include "main.h"
#include "DX12AgilitySDK.h"

extern "C" { __declspec(dllexport) extern const UINT D3D12SDKVersion = 608; }
extern "C" { __declspec(dllexport) extern const char8_t* D3D12SDKPath = u8".\\D3D12\\"; }

int APIENTRY wWinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPWSTR lpCmdLine, _In_ int nCmdShow)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

    if (!DirectX::XMVerifyCPUSupport())
    {
#ifdef _DEBUG
        OutputDebugStringW(L"ERROR: This hardware does not support the required instruction set.\n");
#endif
        return 1;
    }

    D3D12App app(hInstance, nCmdShow);
    app.InitializeWindow();
    app.InitializeD3D12();
    return app.RunMessageLoop();
}
