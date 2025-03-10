#pragma once

#ifndef __BLIZZARD_STORM_HEADER
#define __BLIZZARD_STORM_HEADER

#include <windows.h>
#include <winuser.h>
#include <winsock.h>
#include <ddraw.h>

// Note to self: Linker error => forgot a return value in cpp

// Storm API definition
#ifndef STORMAPI
#define STORMAPI __stdcall
#endif

#ifndef __STORM_SMAX
#define __STORM_SMAX(x,y) (x < y ? y : x)
#endif

#ifndef __STORM_SSIZEMAX
#define __STORM_SSIZEMAX(x,y) (__STORM_SMAX(sizeof(x),sizeof(y)))
#endif

#ifndef __STORM_SMIN
#define __STORM_SMIN(x,y) (x < y ? x : y)
#endif

#ifndef __STORM_SSIZEMIN
#define __STORM_SSIZEMIN(x,y) (__STORM_SMIN(sizeof(x),sizeof(y)))
#endif

typedef struct _WRECT
{
  WORD  left;
  WORD  top;
  WORD  right;
  WORD  bottom;
} WRECT, *PWRECT;

typedef struct _WPOINT
{
  WORD  x;
  WORD  y;
} WPOINT, *PWPOINT;

typedef struct _WSIZE
{
  WORD  cx;
  WORD  cy;
} WSIZE, *PWSIZE;



// Game states
#define GAMESTATE_PRIVATE 0x01
#define GAMESTATE_FULL    0x02
#define GAMESTATE_ACTIVE  0x04
#define GAMESTATE_STARTED 0x08
#define GAMESTATE_REPLAY  0x80


BOOL STORMAPI SNetCreateGame(const char *pszGameName, const char *pszGamePassword, const char *pszGameStatString, DWORD dwGameType, char *GameTemplateData, int GameTemplateSize, int playerCount, char *creatorName, char *a11, int *playerID);
BOOL STORMAPI SNetDestroy();
BOOL STORMAPI SNetEnumProviders(int (STORMAPI *callback)(DWORD, DWORD, DWORD, DWORD), int mincaps);

BOOL STORMAPI SNetEnumGames(int (STORMAPI *callback)(DWORD, DWORD, DWORD), int *hintnextcall);

/*  SNetDropPlayer @ 106
 * 
 *  Drops a player from the current game.
 *  
 *  playerid:     The player ID for the player to be dropped.
 *  flags:        
 *
 *  Returns TRUE if the function was called successfully and FALSE otherwise.
 */
BOOL
STORMAPI
SNetDropPlayer(
      int playerid,
      DWORD flags);

/*  SNetGetGameInfo @ 107
 * 
 *  Retrieves specific game information from Storm, such as name, password,
 *  stats, mode, game template, and players.
 *  
 *  type:         The type of data to retrieve. See GAMEINFO_ flags.
 *  dst:          The destination buffer for the data.
 *  length:       The maximum size of the destination buffer.
 *  byteswritten: The number of bytes written to the destination buffer.
 *
 *  Returns TRUE if the function was called successfully and FALSE otherwise.
 */
BOOL
STORMAPI
SNetGetGameInfo(
    int type,
    void *dst,
    size_t length,
    size_t *byteswritten = NULL);


#define SNGetGameInfo(typ,dst) SNetGetGameInfo(typ, &dst, sizeof(dst))



// Game info fields
#define GAMEINFO_NAME           1
#define GAMEINFO_PASSWORD       2
#define GAMEINFO_STATS          3
#define GAMEINFO_MODEFLAG       4
#define GAMEINFO_GAMETEMPLATE   5
#define GAMEINFO_PLAYERS        6


BOOL STORMAPI SNetGetNumPlayers(int *firstplayerid, int *lastplayerid, int *activeplayers);


typedef struct _CAPS
{
  DWORD dwSize;                 // Size of this structure  // sizeof(CAPS)
  DWORD dwUnk_0x04;             // Some flags?
  DWORD maxmessagesize;         // Size of the packet buffer, must be between 128 and 512
  DWORD dwUnk_0x0C;             // Unknown
  DWORD dwDisplayedPlayerCount; // Displayed player count in the mode selection list
  DWORD dwUnk_0x14;             // some kind of timeout or timer related
  DWORD dwPlayerLatency;        // ... latency?
  DWORD dwPlayerCount;          // the number of players that can participate, must be between 1 and 20
  DWORD dwCallDelay;            // the number of calls before data is sent over the network // between 2 and 8; single player is set to 1
} CAPS, *PCAPS;


BOOL STORMAPI SNetGetPlayerCaps(char playerid, PCAPS playerCaps);

/*  SNetGetPlayerName @ 113
 * 
 *  Retrieves the name of a player given their player ID.
 *  
 *  playerid:     The player's ID.
 *  buffer:       The buffer that will receive the name.
 *  buffersize:   The maximum size of buffer.
 *
 *  Returns TRUE if the function was called successfully and FALSE otherwise.
 */
BOOL
STORMAPI
SNetGetPlayerName(
      int playerid, 
      char *buffer, 
      size_t buffersize);

/*  SNetGetProviderCaps @ 114
 * 
 *  Retrieves network provider capacity information.
 *  
 *  providerCaps: A pointer to a CAPS structure that will receive the information.
 *
 *  Returns TRUE if the function was called successfully and FALSE otherwise.
 */
BOOL 
STORMAPI 
SNetGetProviderCaps(
      PCAPS providerCaps);

/*  SNetGetTurnsInTransit @ 115
 * 
 *  Retrieves the number of turns (buffers) that have been queued
 *  before sending them over the network.
 *  
 *  turns: A pointer to an integer that will receive the value.
 *
 *  Returns TRUE if the function was called successfully and FALSE otherwise.
 */
BOOL
STORMAPI
SNetGetTurnsInTransit(
      int *turns);


BOOL STORMAPI SNetInitializeDevice(int a1, int a2, int a3, int a4, int *a5);

// Network provider structures
typedef struct _client_info
{
  DWORD dwSize; // 60
  char  *pszName;
  char  *pszVersion;
  DWORD dwProduct;
  DWORD dwVerbyte;
  DWORD dwUnk5;
  DWORD dwMaxPlayers;
  DWORD dwUnk7;
  DWORD dwUnk8;
  DWORD dwUnk9;
  DWORD dwUnk10; // 0xFF
  char  *pszCdKey;
  char  *pszCdOwner;
  DWORD dwIsShareware;
  DWORD dwLangId;
} client_info;

typedef struct _user_info
{
  DWORD dwSize; // 16
  char  *pszPlayerName;
  char  *pszUnknown;
  DWORD dwUnknown;
} user_info;

typedef struct _battle_info
{
  DWORD dwSize;   // 92
  DWORD dwUnkType;
  HWND  hFrameWnd;
  void  *pfnBattleGetResource;
  void  *pfnBattleGetErrorString;
  void  *pfnBattleMakeCreateGameDialog;
  void  *pfnBattleUpdateIcons;
  DWORD dwUnk_07;
  void  *pfnBattleErrorDialog;
  void  *pfnBattlePlaySound;
  DWORD dwUnk_10;
  void  *pfnBattleGetCursorLink;
  DWORD dwUnk_12;
  void  *pfnUnk_13;
  DWORD dwUnk_14;
  void  *pfnBattleMakeProfileDialog;
  char  *pszProfileStrings;
  void  *pfnBattleDrawProfileInfo;
  void  *pfnUnk_18;
  DWORD dwUnk_19;
  void  *pfnUnk_20;
  void  *pfnUnk_21;
  void  *pfnBattleSetLeagueName;
} battle_info;

typedef struct _module_info
{
  DWORD dwSize; // 20
  char  *pszVersionString;
  char  *pszModuleName;
  char  *pszMainArchive;
  char  *pszPatchArchive;
} module_info;

typedef struct _game
{
  DWORD     dwIndex;
  DWORD     dwGameState;
  DWORD     dwUnk_08;
  SOCKADDR  saHost;
  DWORD     dwUnk_1C;
  DWORD     dwTimer;
  DWORD     dwUnk_24;
  char      szGameName[128];
  char      szGameStatString[128];
  _game     *pNext;
  void      *pExtra;
  DWORD     dwExtraBytes;
  DWORD     dwProduct;
  DWORD     dwVersion;
} game;

typedef struct _storm_head
{
  WORD wChecksum;
  WORD wLength;
  WORD wSent;
  WORD wReceived;
  BYTE bCommandClass;
  BYTE bCommandType;
  BYTE bPlayerId;
  BYTE bFlags;
} storm_head;


// Traffic flags
#define STRAFFIC_NORMAL 0
#define STRAFFIC_VERIFY 1
#define STRAFFIC_RESEND 2
#define STRAFFIC_REPLY  4


/*  SNetInitializeProvider @ 117
 * 
 *  Initializes a provider by storing the provider callbacks, and calling
 *  spiInitialize() using the parameters passed to this function.
 *  Note: The use of the parameters is determined by the network
 *  module.
 *  
 *  providerName:     The provider's identifier. Example: 'TENB' (BNET).
 *  gameClientInfo:   A pointer to a clientInfo structure containing 
 *                    information about the game client.
 *  userData:         A pointer to a userInfo structure containing information 
 *                    about the player.
 *  bnCallbacks:      A pointer to a battleInfo structure containing callbacks
 *                    and other information that is specific to Battle.net.
 *  moduleData:       A pointer to a moduleInfo structure containing the
 *                    executable information and paths to MPQ archives.
 *
 *  Returns TRUE if the function was called successfully and FALSE otherwise.
 */
BOOL
STORMAPI
SNetInitializeProvider(
      DWORD       providerName, 
      client_info  *gameClientInfo,
      user_info    *userData,
      battle_info  *bnCallbacks,
      module_info  *moduleData);


BOOL STORMAPI SNetJoinGame(int id, char *gameName, char *gamePassword, char *playerName, char *userStats, int *playerid);

/*  SNetLeaveGame @ 119
 * 
 *  Notifies Storm that the player has left the game. Storm will
 *  notify all connected peers through the network provider.
 *  
 *  type: The leave type. It doesn't appear to be important, no documentation available.
 *
 *  Returns TRUE if the function was called successfully and FALSE otherwise.
 */
BOOL
STORMAPI
SNetLeaveGame(
    int type);

BOOL STORMAPI SNetPerformUpgrade(DWORD *upgradestatus);
BOOL STORMAPI SNetReceiveMessage(int *senderplayerid, char **data, int *databytes);
BOOL STORMAPI SNetReceiveTurns(int a1, int arraysize, char **arraydata, unsigned int *arraydatabytes, DWORD *arrayplayerstatus);

// Values for arrayplayerstatus
#define SNET_PS_OK             0
#define SNET_PS_WAITING        2
#define SNET_PS_NOTRESPONDING  3
#define SNET_PS_UNKNOWN        default


// Event structure
typedef struct _s_evt
{
  DWORD dwFlags;
  int   dwPlayerId;
  void  *pData;
  DWORD dwSize;
} S_EVT, *PS_EVT;


// @TODO: "type" is unknown.
HANDLE STORMAPI SNetRegisterEventHandler(int type, void (STORMAPI *sEvent)(PS_EVT));

int  STORMAPI SNetSelectGame(int a1, int a2, int a3, int a4, int a5, int *playerid);

/*  SNetSendMessage @ 127
 * 
 *  Sends a message to a player given their player ID. Network message
 *  is sent using class 01 and is retrieved by the other client using
 *  SNetReceiveMessage().
 *  
 *  playerID:   The player index of the player to receive the data.
 *              Conversely, this field can be one of the following constants:
 *                  SNPLAYER_ALL      | Sends the message to all players, including oneself.
 *                  SNPLAYER_OTHERS   | Sends the message to all players, except for oneself.
 *  data:       A pointer to the data.
 *  databytes:  The amount of bytes that the data pointer contains.
 *
 *  Returns TRUE if the function was called successfully and FALSE otherwise.
 */
BOOL
STORMAPI
SNetSendMessage(
      int    playerID,
      void   *data,
      size_t databytes);


// Macro values to target specific players
#define SNPLAYER_ALL    -1
#define SNPLAYER_OTHERS -2


/*  SNetSendTurn @ 128
 * 
 *  Sends a turn (data packet) to all players in the game. Network data
 *  is sent using class 02 and is retrieved by the other client using
 *  SNetReceiveTurns().
 *  
 *  data:       A pointer to the data.
 *  databytes:  The amount of bytes that the data pointer contains.
 *
 *  Returns TRUE if the function was called successfully and FALSE otherwise.
 */
BOOL
STORMAPI
SNetSendTurn(
      char    *data,
      size_t  databytes);

/*  SNetSetGameMode @ 130
 * 
 *  Set's the game's mode flags, notifying the network
 *  provider that the state of the game has changed.
 *  For example: notifies Battle.net when the game is
 *  full.
 *  
 *  You should first call SNetGetGameInfo to retrieve
 *  the existing mode flags.
 *
 *  modeFlags:  The new flags for the game mode.
 *                  GAMESTATE_PRIVATE     | The game is passworded.
 *                  GAMESTATE_FULL        | The game is full.
 *                  GAMESTATE_ACTIVE      | The game is available.
 *                  GAMESTATE_STARTED     | The game is in progress.
 *                  GAMESTATE_REPLAY      | The game is a replay.
 *  makePublic: Used to make the game a public game, removing the GAMESTATE_PRIVATE flag.
 *
 *  Returns TRUE if the function was called successfully and FALSE otherwise.
 */
BOOL
STORMAPI
SNetSetGameMode(
      DWORD modeFlags,
      bool makePublic = false);

#define SNMakeGamePublic() SNetSetGameMode( (DWORD mode, SNetGetGameInfo(GAMEINFO_MODEFLAGS, &mode, 4), mode), true)

BOOL STORMAPI SNetEnumGamesEx(int a1, int a2, int (__fastcall *callback)(DWORD, DWORD, DWORD), int *hintnextcall);
BOOL STORMAPI SNetSendServerChatCommand(const char *command);

BOOL STORMAPI SNetDisconnectAll(DWORD flags);
BOOL STORMAPI SNetCreateLadderGame(const char *pszGameName, const char *pszGamePassword, const char *pszGameStatString, DWORD dwGameType, DWORD dwGameLadderType, DWORD dwGameModeFlags, char *GameTemplateData, int GameTemplateSize, int playerCount, char *creatorName, char *a11, int *playerID);

#define SNET_GAME_RESULT_WIN        1
#define SNET_GAME_RESULT_LOSS       2
#define SNET_GAME_RESULT_DRAW       3
#define SNET_GAME_RESULT_DISCONNECT 4

BOOL STORMAPI SNetReportGameResult(unsigned a1, int size, int *results, const char* headerInfo, const char* detailInfo);

int  STORMAPI SNetSendLeagueCommand(char *cmd, char *callback);
int  STORMAPI SNetSendReplayPath(int a1, int a2, char *replayPath);
int  STORMAPI SNetGetLeagueName(int leagueID);
BOOL STORMAPI SNetGetPlayerNames(char **names);
int  STORMAPI SNetLeagueLogout(char *bnetName);
int  STORMAPI SNetGetLeaguePlayerName(char *curPlayerLeageName, size_t nameSize);

HGDIOBJ STORMAPI SDlgDefDialogProc(HWND hDlg, signed int DlgType, HDC textLabel, HWND hWnd);

HANDLE STORMAPI SDlgDialogBoxIndirectParam(HMODULE hModule, LPCSTR lpName, HWND hWndParent, LPVOID lpParam, LPARAM lParam);

BOOL STORMAPI SDlgEndDialog(HWND hDlg, HANDLE nResult);

BOOL STORMAPI SDlgSetControlBitmaps(HWND parentwindow, int *id, int a3, char *buffer2, char *buffer, int flags, int mask);

/*
// lpCursorName can only be IDC_ARROW 
BOOL STORMAPI SDlgSetSystemCursor(void *lpSrcBuffer, void *p_a2, LPSIZE lpSize, LPCSTR lpCursorName);
*/

BOOL STORMAPI SDlgBltToWindowE(HWND hWnd, HRGN a2, char *a3, int a4, void *buffer, RECT *rct, SIZE *size, int a8, int a9, DWORD rop);
BOOL STORMAPI SDlgSetBitmapE(HWND hWnd, int a2, char *src, int mask1, int flags, int a6, int a7, int width, int a9, int mask2);

int  STORMAPI Ordinal224(int a1);

BOOL STORMAPI SFileCloseArchive(HANDLE hArchive);
BOOL STORMAPI SFileCloseFile(HANDLE hFile);

BOOL STORMAPI SFileDdaBeginEx(HANDLE directsound, DWORD flags, DWORD mask, unsigned __int32 lDistanceToMove, signed __int32 volume, signed int a6, int a7);
BOOL STORMAPI SFileDdaDestroy();
BOOL STORMAPI SFileDdaEnd(HANDLE directsound);
BOOL STORMAPI SFileDdaGetPos(HANDLE directsound, int a2, int a3);

BOOL STORMAPI SFileDdaInitialize(HANDLE directsound);
BOOL STORMAPI SFileDdaSetVolume(HANDLE directsound, signed int bigvolume, signed int volume);
BOOL STORMAPI SFileDestroy();

BOOL STORMAPI SFileGetFileArchive(HANDLE hFile, HANDLE archive);
LONG STORMAPI SFileGetFileSize(HANDLE hFile, LPDWORD lpFileSizeHigh);
BOOL STORMAPI SFileOpenArchive(const char *szMpqName, DWORD dwPriority, DWORD dwFlags, HANDLE *phMpq);

// values for dwFlags
enum MPQFlags
{
  MPQ_NO_LISTFILE       = 0x0010,
  MPQ_NO_ATTRIBUTES     = 0x0020,
  MPQ_FORCE_V1          = 0x0040,
  MPQ_CHECK_SECTOR_CRC  = 0x0080
};


BOOL STORMAPI SFileOpenFile(const char *filename, HANDLE *phFile);
BOOL STORMAPI SFileOpenFileEx(HANDLE hMpq, const char *szFileName, DWORD dwSearchScope, HANDLE *phFile);

// values for dwSearchScope
enum SFileFlags
{
  SFILE_FROM_MPQ      = 0x00000000,
  SFILE_FROM_ABSOLUTE = 0x00000001,
  SFILE_FROM_RELATIVE = 0x00000002,
  SFILE_FROM_DISK     = 0x00000004
};

BOOL STORMAPI SFileReadFile(HANDLE hFile, void *buffer, DWORD nNumberOfBytesToRead, DWORD *read, LONG lpDistanceToMoveHigh);

void STORMAPI SFileSetLocale(LCID lcLocale);

// mode:    0 - Silent (callback is NULL)
//          1 - Application Defined
//          2 - Handled by storm (callback is NULL)
// BOOL STORMAPI callback(const char *pszFilename, DWORD dwErrCode, DWORD dwErrCount)
BOOL STORMAPI SFileSetIoErrorMode(DWORD mode, BOOL (STORMAPI *callback)(const char*,DWORD,DWORD) );

BOOL STORMAPI SFileGetArchiveName(HANDLE hArchive, char *name, int length);
BOOL STORMAPI SFileGetFileName(HANDLE hFile, char *buffer, int length);

BOOL STORMAPI SFileLoadFile(char *filename, void *buffer, int buffersize, int a4, int a5);
BOOL STORMAPI SFileUnloadFile(HANDLE hFile);
BOOL STORMAPI SFileLoadFileEx(void *hArchive, char *filename, int a3, int a4, int a5, DWORD searchScope, struct _OVERLAPPED *lpOverlapped);

// Options are DWORD except for #6
// 1: [TRUE|FALSE] - If true, reports resource leaks (SErrReportResourceLeak/SErrReportNamedResourceLeak) to the attached debugger instead of a message box.
// 2: This option is unused.
// 3: [TRUE|FALSE] - If true, reports general memory leaks to the attached debugger instead of a message box.
// 4: This option is unused.
// 5: [TRUE|FALSE] - If true, reports log messages and log dumps to the attached debugger.
// 6: { DWORD blocks_allocated; DWORD blocks_freed; } Used to determine the amount of memory/heap blocks that have been allocated and freed by storm.
//    Can also be used for custom allocations outside of storm.
//
//BOOL STORMAPI StormGetOption(int type, void *pValue, size_t *pSize);
//BOOL STORMAPI StormSetOption(int type, void *pValue, size_t size);

BOOL STORMAPI SBltROP3(void *lpDstBuffer, void *lpSrcBuffer, int srcDrawWidth, int srcDrawHeight, int dstWidth, int srcWidth, int a7, DWORD rop);
BOOL STORMAPI SBltROP3Clipped(void *lpDstBuffer, RECT *lpDstRect, POINT *lpDstPt, int a4, void *lpSrcBuffer, RECT *lpSrcRect, POINT *lpSrcPt, int a8, int a9, DWORD rop);

#define SBMP_DEFAULT  0
#define SBMP_BMP      1
#define SBMP_PCX      2
#define SBMP_TGA      3


/*  SBmpDecodeImage @ 321
 * 
 *  Decodes an image that has already been loaded into a buffer.
 *  
 *  dwImgType:        Optional, the image type. See SBMP_ macros.
 *  pSrcBuffer:       A pointer to the source buffer.
 *  dwSrcBuffersize:  The size of the data in the source buffer.
 *  pPalette:         An optional buffer that receives the image palette.
 *  pDstBuffer:       A buffer that receives the image data.
 *  dwDstBuffersize:  The size of the specified image buffer. If the size of the 
 *                    destination buffer is 0, then the destination buffer is not used.
 *  pdwWidth:         An optional variable that receives the image width.
 *  pdwHeight:        An optional variable that receives the image height.
 *  pdwBpp:           An optional variable that receives the image bits per pixel.
 *
 *  Returns TRUE if the image was supported and decoded correctly, FALSE otherwise.
 */
BOOL 
STORMAPI 
SBmpDecodeImage(
    DWORD        dwImgType,
    void         *pSrcBuffer,
    DWORD        dwSrcBuffersize,
    PALETTEENTRY *pPalette        = NULL,
    void         *pDstBuffer      = NULL,
    DWORD        dwDstBuffersize  = 0,
    DWORD        *pdwWidth        = NULL,
    DWORD        *pdwHeight       = NULL,
    DWORD        *pdwBpp          = NULL);


/*  SBmpLoadImage @ 323
 * 
 *  Load an image from an available archive into a buffer.
 *  
 *  pszFileName:  The name of the graphic in an active archive.
 *  pPalette:     An optional buffer that receives the image palette.
 *  pBuffer:      A buffer that receives the image data.
 *  dwBuffersize: The size of the specified image buffer.
 *  pdwWidth:     An optional variable that receives the image width.
 *  pdwHeight:    An optional variable that receives the image height.
 *  pdwBpp:       An optional variable that receives the image bits per pixel.
 *
 *  Returns TRUE if the image was supported and loaded correctly, FALSE otherwise.
 */
BOOL
STORMAPI
SBmpLoadImage(
    const char   *pszFileName,
    PALETTEENTRY *pPalette    = NULL,
    void         *pBuffer     = NULL,
    DWORD        dwBuffersize = 0,
    DWORD        *pdwWidth    = NULL,
    DWORD        *pdwHeight   = NULL,
    DWORD        *pdwBpp      = NULL);

/*  SBmpSaveImage @ 324
 * 
 *  Save an image from a buffer to a file. The image format is determined 
 *  from the filename and is either .gif, .pcx, .tga, or .bmp being the default.
 *  
 *  pszFileName:  The name of the file to create.
 *  pPalette:     A pointer to a palette array containing 256 entries.
 *  pBuffer:      A buffer containing the image data.
 *  pdwWidth:     The width of the image.
 *  pdwHeight:    The height of the image.
 *  pdwBpp:       The bits per pixel.
 *
 *  Returns TRUE if the image was saved correctly, FALSE otherwise.
 */
BOOL
STORMAPI
SBmpSaveImage(
    const char    *pszFileName,
    PALETTEENTRY  *pPalette,
    void          *pBuffer,
    DWORD         dwWidth,
    DWORD         dwHeight,
    DWORD         dwBpp = 8);


HANDLE STORMAPI SBmpAllocLoadImage(const char *fileName, PALETTEENTRY *palette, void **buffer, int *width, int *height, int unused6, int unused7, void *(STORMAPI *allocFunction)(DWORD));

BOOL STORMAPI SCodeCompile(char *directives1, char *directives2, char *loopstring, unsigned int maxiterations, unsigned int flags, HANDLE handle);
BOOL STORMAPI SCodeDelete(HANDLE handle);

int  STORMAPI SCodeExecute(HANDLE handle, int a2);

BOOL STORMAPI SDrawAutoInitialize(HINSTANCE hInst, LPCSTR lpClassName, LPCSTR lpWindowName, WNDPROC pfnWndProc, int nMode, int nWidth, int nHeight, int nBits);


/*  SDrawCaptureScreen @ 342
 * 
 *  Saves a screenshot from the primary surface being handled by Storm.
 *  
 *  pszOutput: The name of the output file. The save format is automatically set by the extension.
 *             The extensions supported are .gif, .pcx, .tga, and .bmp. It will write a bitmap by default.
 *
 *  Returns TRUE if successful and FALSE otherwise.
 */
BOOL 
STORMAPI 
SDrawCaptureScreen(
    const char *pszOutput);


/*  SDrawGetFrameWindow @ 346
 *
 *  Retrieves the window handle that was specified in
 *  SDrawManualInitialize or created in SDrawAutoInitialize.
 *  
 *  sdraw_framewindow: Optional variable that receives the returned handle.
 *
 *  Returns the handle of the window.
 */
HWND
STORMAPI
SDrawGetFrameWindow(
    HWND *sdraw_framewindow = NULL);


/*  SDrawGetObjects @ 347
 *  
 *  Retrieves the object information that was initialized using
 *  SDrawManualInitialize or SDrawAutoInitialize.
 *  
 *  ddInterface:    The DirectDraw interface.
 *  primarySurface: The primary DirectDraw surface.
 *  surface2:       A second unknown surface.
 *  surface3:       A third unknown surface.
 *  backSurface:    The back DirectDraw surface.
 *  ddPalette:      The DirectDraw palette.
 *  hPalette:       The palette handle.
 *
 *  Returns FALSE if the direct draw interface has not been initialized.
 */
BOOL
STORMAPI
SDrawGetObjects(
    LPDIRECTDRAW        *ddInterface    = NULL,
    LPDIRECTDRAWSURFACE *primarySurface = NULL,
    LPDIRECTDRAWSURFACE *surface2       = NULL,
    LPDIRECTDRAWSURFACE *surface3       = NULL,
    LPDIRECTDRAWSURFACE *backSurface    = NULL,
    LPDIRECTDRAWPALETTE *ddPalette      = NULL,
    HPALETTE            *hPalette       = NULL);


/*  SDrawGetScreenSize @ 348
 *  
 *  Obtains information for the current screen resolution.
 *
 *  pdwWidth:   Optional variable that receives the screen width.
 *  pdwHeight:  Optional variable that receives the screen height.
 *  pdwBpp:     Optional variable that receives the bits per pixel.
 *
 *  Returns FALSE if no variables were specified.
 */
BOOL
STORMAPI
SDrawGetScreenSize(
    DWORD *pdwWidth,
    DWORD *pdwHeight,
    DWORD *pdwBpp);


// undefined
BOOL STORMAPI SDrawLockSurface(int surfacenumber, RECT *lpDestRect, void **lplpSurface, int *lpPitch, int arg_unused);


/*  SDrawManualInitialize @ 351
 *  
 *  Sets the DirectDraw variables to be referenced in Storm.
 *
 *  hWnd:           The handle of the DirectDraw window.
 *  ddInterface:    The DirectDraw interface.
 *  primarySurface: The first and primary surface.
 *  surface2:       A second surface. Behaviour not completely known.
 *  surface3:       A third surface. Behaviour not completely known.
 *  backSurface:    The fourth and final surface. The back surface.
 *  ddPalette:      The DirectDraw palette if the application requires it.
 *  hPalette:       The palette handle that belongs to the window.
 *                  If this is NULL and ddPalette is specified, then it
 *                  will be created automatically. A palette can be created
 *                  using the CreatePalette WinAPI function.
 *
 *  Returns FALSE if no variables were specified.
 */
BOOL
STORMAPI
SDrawManualInitialize(
     HWND hWnd                           = NULL,
     LPDIRECTDRAW ddInterface            = NULL,
     LPDIRECTDRAWSURFACE primarySurface  = NULL,
     LPDIRECTDRAWSURFACE surface2        = NULL,
     LPDIRECTDRAWSURFACE surface3        = NULL,
     LPDIRECTDRAWSURFACE backSurface     = NULL,
     LPDIRECTDRAWPALETTE ddPalette       = NULL,
     HPALETTE hPalette                   = NULL);


/*  SDrawPostClose @ 353
 *
 *  Posts a WM_QUIT message to the active drawing window specified 
 *  in SDrawManualInitialize or created in SDrawAutoInitialize.
 *
 *  Returns TRUE if successful and FALSE otherwise.
 */
BOOL
STORMAPI
SDrawPostClose();


// undefined
//BOOL STORMAPI SDrawRealizePalette();

BOOL STORMAPI SDrawUnlockSurface(int surfacenumber, void *lpSurface, int a3, RECT *lpRect);
BOOL STORMAPI SDrawUpdatePalette(unsigned int firstentry, unsigned int numentries, PALETTEENTRY *pPalEntries, int a4);

BOOL STORMAPI SEvtDispatch(DWORD dwMessageID, DWORD dwFlags, int type, PS_EVT pEvent);

BOOL STORMAPI SGdiDeleteObject(HANDLE handle);

BOOL STORMAPI SGdiExtTextOut(int a1, int a2, int a3, int a4, unsigned int a8, signed int a6, signed int a7, const char *pszString, unsigned int arg20);
BOOL STORMAPI SGdiImportFont(HGDIOBJ handle, int windowsfont);

BOOL STORMAPI SGdiSelectObject(int handle);
BOOL STORMAPI SGdiSetPitch(int pitch);

BOOL STORMAPI Ordinal393(char *pszString, int, int);


/*  SMemAlloc @ 401
 *  
 *  Allocates a block of memory. This block is different
 *  from the standard malloc by including a header containing
 *  information about the block. 
 *
 *  amount:       The amount of memory to allocate, in bytes.
 *  logfilename:  The name of the file or object that this call belongs to.
 *  logline:      The line in the file or one of the SLOG_ macros.
 *  defaultValue: The default value of a byte in the allocated memory.
 *
 *  Returns a pointer to the allocated memory. This pointer does NOT include
 *  the additional storm header.
 */
void*
STORMAPI
SMemAlloc(
    size_t amount,
    char  *logfilename,
    int   logline,
    char  defaultValue = 0);

#define SMAlloc(amount) SMemAlloc((amount), __FILE__, __LINE__)


/*  SMemFree @ 403
 *  
 *  Frees a block of memory that was created using SMemAlloc, 
 *  includes the log file and line for debugging purposes.
 *
 *  location:     The memory location to be freed.
 *  logfilename:  The name of the file or object that this call belongs to.
 *  logline:      The line in the file or one of the SLOG_ macros.
 *  defaultValue: 
 *
 *  Returns TRUE if the call was successful and FALSE otherwise.
 */
BOOL
STORMAPI
SMemFree(
    void *location,
    char *logfilename,
    int  logline,
    char defaultValue = 0);

#define SMFree(loc) SMemFree((loc), __FILE__, __LINE__)


/*  SMemReAlloc @ 405
 *  
 *  Reallocates a block of memory that was created using SMemAlloc, 
 *  includes the log file and line for debugging purposes.
 *
 *  location:     The memory location to be re-allocated. If this parameter
 *                is NULL, then SMemAlloc is called with the remaining parameters.
 *  amount:       The amount of memory to re-allocate.
 *  logfilename:  The name of the file or object that this call belongs to.
 *  logline:      The line in the file or one of the SLOG_ macros.
 *  defaultValue: 
 *
 *  Returns a pointer to the re-allocated memory. This pointer does NOT include
 *  the additional storm header.
 */
void*
STORMAPI
SMemReAlloc(
    void    *location,
    size_t  amount,
    char    *logfilename,
    int     logline,
    char    defaultValue = 0);

#define SMReAlloc(loc,s) SMemReAlloc((loc),(s), __FILE__, __LINE__)

// Can be provided instead of logline/__LINE__ parameter to indicate different errors.
#define SLOG_EXPRESSION    0
#define SLOG_FUNCTION     -1
#define SLOG_OBJECT       -2
#define SLOG_HANDLE       -3
#define SLOG_FILE         -4
#define SLOG_EXCEPTION    -5


BOOL STORMAPI SRegLoadData(const char *keyname, const char *valuename, int size, LPBYTE lpData, BYTE flags, LPDWORD lpcbData);
BOOL STORMAPI SRegLoadString(const char *keyname, const char *valuename, BYTE flags, char *buffer, size_t buffersize);
BOOL STORMAPI SRegLoadValue(const char *keyname, const char *valuename, BYTE flags, int *value);
BOOL STORMAPI SRegSaveData(const char *keyname, const char *valuename, int size, BYTE *lpData, DWORD cbData);
BOOL STORMAPI SRegSaveString(const char *keyname, const char *valuename, BYTE flags, char *string);
BOOL STORMAPI SRegSaveValue(const char *keyname, const char *valuename, BYTE flags, DWORD result);

BOOL STORMAPI SRegDeleteValue(const char *keyname, const char *valuename, BYTE flags);

// Flags for SReg functions

// Default behaviour checks both HKEY_LOCAL_MACHINE and HKEY_CURRENT_USER
// relative to the "Software\\Blizzard Entertainment\\" key in both hives.
#define SREG_NONE                   0x00000000
#define SREG_EXCLUDE_LOCAL_MACHINE  0x00000001  // excludes checking the HKEY_LOCAL_MACHINE hive
#define SREG_BATTLE_NET             0x00000002  // sets the relative key to "Software\\Battle.net\\" instead
#define SREG_EXCLUDE_CURRENT_USER   0x00000004  // excludes checking the HKEY_CURRENT_USER hive
#define SREG_ABSOLUTE               0x00000010  // specifies that the key is not a relative key

BOOL STORMAPI STransBlt(void *lpSurface, int x, int y, int width, HANDLE hTrans);
BOOL STORMAPI STransBltUsingMask(void *lpDest, void *lpSource, int pitch, int width, HANDLE hTrans);

BOOL STORMAPI STransDelete(HANDLE hTrans);

BOOL STORMAPI STransDuplicate(HANDLE hTransSource, HANDLE hTransDest);
BOOL STORMAPI STransIntersectDirtyArray(HANDLE hTrans, char * dirtyarraymask, unsigned flags, HANDLE * phTransResult);
BOOL STORMAPI STransInvertMask(HANDLE hTrans, HANDLE * phTransResult);

BOOL STORMAPI STransSetDirtyArrayInfo(int width, int height, int depth, int bits);

BOOL STORMAPI STransPointInMask(HANDLE hTrans, int x, int y); // Name is a pure guess
BOOL STORMAPI STransCombineMasks(HANDLE hTransA, HANDLE hTransB, int left, int top, int flags, HANDLE * phTransResult);

BOOL STORMAPI STransCreateE(void *pBuffer, int width, int height, int bpp, int a5, int bufferSize, HANDLE *phTransOut);

BOOL STORMAPI SVidDestroy();
BOOL STORMAPI SVidGetSize(HANDLE video, int width, int height, int zero);
BOOL STORMAPI SVidInitialize(HANDLE video);
BOOL STORMAPI SVidPlayBegin(char *filename, int arg4, int a3, int a4, int a5, int a6, HANDLE* video);

BOOL STORMAPI SVidPlayContinueSingle(HANDLE video, int a2, int a3);
BOOL STORMAPI SVidPlayEnd(HANDLE video);

/* SErrDisplayError @ 461
 *
 * Displays a formatted error message. The message is detailed and flexible for many applications.
 * The message will be different if there is a debugger attached. Will typically terminate the application
 * unless the option to continue is given.
 *
 *  dwErrMessage:   The error code. See SErrGetLastError and GetLastError.
 *  logfilename:    The name of the file or object that this call belongs to.
 *  logline:        The line in the file or one of the SLOG_ macros.
 *  message:        A message or expression with additional information.
 *  allowOption:    If TRUE, allows the user the option to continue execution, otherwise the program will terminate.
 *  exitCode:       The exit code used for program termination.
 *
 *  Returns TRUE if the user chose to continue execution, FALSE otherwise.
 */
BOOL
STORMAPI
SErrDisplayError(
    DWORD dwErrMsg,
    const char *logfilename,
    int logline,
    const char *message = NULL,
    BOOL allowOption = FALSE,
    int exitCode = 1);

#define SAssert(x) { if ( !(x) ) SErrDisplayError(STORM_ERROR_ASSERTION, __FILE__, __LINE__, #x) }

#define SEDisplayError(err) SErrDisplayError(e, __FILE__, __LINE__)

/*  SErrGetErrorStr @ 462
 *  
 *  Retrieves a string that describes the specified error code for
 *  the system, Storm, DirectDraw, or DirectSound.
 *
 *  dwErrCode:    The error code to look up.
 *  buffer:       The destination buffer to receive the string.
 *  bufferchars:  The size of the destination buffer.
 *
 *  Returns TRUE if the call was successful and FALSE otherwise.
 */
BOOL
STORMAPI
SErrGetErrorStr(
    DWORD dwErrCode,
    char *buffer,
    size_t bufferchars);

#define SEGetErrorStr(e,b) SErrGetErrorStr(e,b,sizeof(b))


/*  SErrGetLastError @ 463
 *  
 *  Retrieves the last error that was specifically
 *  set for the Storm library.
 *
 *  Returns the last error set within the Storm library.
 */
DWORD
STORMAPI
SErrGetLastError();


// Registers a module as a message source for SErrGetErrorStr, always returns TRUE
// groupID is a group in a MessageTable entry for example in STORM_ERROR_BAD_ARGUMENT 0x85100065, 0x510 is the group.
// BOOL STORMAPI SErrRegisterMessageSource(WORD groupID, HMODULE hSourceModule, int a3)


/*  SErrSetLastError @ 465
 *  
 *  Sets the last error for the Storm library and the Kernel32 library.
 *
 *  dwErrCode:  The error code that will be set.
 */
void
STORMAPI
SErrSetLastError(
    DWORD dwErrCode = NO_ERROR);

// 
// void STORMAPI SErrReportNamedResourceLeak(const char *pszMsg, const char *pszSubMsg = nullptr)
// void STORMAPI SErrReportResourceLeak(const char *pszMsg)

void STORMAPI SErrSuppressErrors(BOOL suppressErrors);

// Values for dwErrCode
#define STORM_ERROR_ASSERTION                    0x85100000
#define STORM_ERROR_BAD_ARGUMENT                 0x85100065
#define STORM_ERROR_GAME_ALREADY_STARTED         0x85100066
#define STORM_ERROR_GAME_FULL                    0x85100067
#define STORM_ERROR_GAME_NOT_FOUND               0x85100068
#define STORM_ERROR_GAME_TERMINATED              0x85100069
#define STORM_ERROR_INVALID_PLAYER               0x8510006a
#define STORM_ERROR_NO_MESSAGES_WAITING          0x8510006b
#define STORM_ERROR_NOT_ARCHIVE                  0x8510006c
#define STORM_ERROR_NOT_ENOUGH_ARGUMENTS         0x8510006d
#define STORM_ERROR_NOT_IMPLEMENTED              0x8510006e
#define STORM_ERROR_NOT_IN_ARCHIVE               0x8510006f
#define STORM_ERROR_NOT_IN_GAME                  0x85100070
#define STORM_ERROR_NOT_INITIALIZED              0x85100071
#define STORM_ERROR_NOT_PLAYING                  0x85100072
#define STORM_ERROR_NOT_REGISTERED               0x85100073
#define STORM_ERROR_REQUIRES_CODEC1              0x85100074
#define STORM_ERROR_REQUIRES_CODEC2              0x85100075
#define STORM_ERROR_REQUIRES_CODEC3              0x85100076
#define STORM_ERROR_REQUIRES_UPGRADE             0x85100077
#define STORM_ERROR_STILL_ACTIVE                 0x85100078
#define STORM_ERROR_VERSION_MISMATCH             0x85100079
#define STORM_ERROR_MEM_NOT_ALLOCATED            0x8510007a
#define STORM_ERROR_MEM_CORRUPTED                0x8510007b
#define STORM_ERROR_MEM_INVALID                  0x8510007c
#define STORM_ERROR_MEM_MANAGER_NOT_INITIALIZED  0x8510007d
#define STORM_ERROR_MEM_NOT_FREED                0x8510007e
#define STORM_ERROR_RESOURCES_NOT_RELEASED       0x8510007f
#define STORM_ERROR_OUT_OF_BOUNDS                0x85100080
#define STORM_ERROR_NULL_POINTER                 0x85100081
#define STORM_ERROR_CDKEY_MISMATCH               0x85100082
#define STORM_ERROR_FILE_CORRUPTED               0x85100083
#define STORM_ERROR_FATAL                        0x85100084
#define STORM_ERROR_GAMETYPE_UNAVAILABLE         0x85100085


/*  SMemCopy @ 491
 *  
 *  Copies a block of memory from source to destination.
 *  This function immediately calls memcpy. See online documentation
 *  of memcpy for more details.
 *
 *  dest:   The destination buffer.
 *  source: The source buffer.
 *  size:   The number of bytes to copy.
 */
void
STORMAPI
SMemCopy(
    void *dest, 
    const void *source, 
    size_t size);

#define SMCopy(d,s) ( SMemCopy(d, s, __STORM_SSIZEMIN(s,d)) )

/*  SMemFill @ 492
 *  
 *  Fills a block of memory with the specified character.
 *  This function immediately calls memset. See online documentation
 *  of memset for more details.
 *
 *  dest:   The destination buffer.
 *  source: The size of the destination buffer.
 *  size:   The format to use.
 */
void
STORMAPI
SMemFill(
    void *location,
    size_t length,
    char fillWith = 0);

#define SMFill(l,f) (SMemFill(l, sizeof(l), f))

/*  SMemZero @ 494
 *  
 *  Fills a block of memory with the integer 0x00 (Zero).
 *
 *  location: The location to write at.
 *  length:   The amount of bytes to write.
 */
void 
STORMAPI 
SMemZero(
    void *location,
    size_t length);

#define SMZero(l) (SMemZero(l, sizeof(l)))


int   STORMAPI SMemCmp(void *location1, void *location2, DWORD size);

#define SMCmp(l,x) ( SMemCmp(l, x, __STORM_SSIZEMIN(x,l)) )

/*  SStrCopy @ 501
 *  
 *  Copies a string from src to dest (including NULL terminator)
 *  until the max_length is reached.
 *
 *  dest:         The destination array.
 *  src:          The source array.
 *  max_length:   The maximum length of dest.
 *
 *  Returns the number of characters copied.
 */
int
STORMAPI
SStrCopy(
    char *dest, 
    const char *src,
    int max_length = 0x7FFFFFFF);

#define SSCopy(d,s) (SStrCopy(d, s, sizeof(d)))

#define STORM_HASH_ABSOLUTE 1

/*  SStrHash @ 502
 *  
 *  Creates a simple hash for the string. This function
 *  should NOT be used for sensitive information.
 *
 *  string:   The input string.
 *  flags:    If STORM_HASH_ABSOLUTE is set then this
              function uses the absolute string, otherwise
              it will convert backslashes to forward
              slashes and some other processing.
 *  seed:     The hash seed. If this value is 0 then the
 *            default value 0x7FED7FED will be used.
 *
 *  Returns the 32-bit hash of the string.
 */
DWORD
STORMAPI
SStrHash(
      const char *string,
      DWORD flags = 0,
      DWORD Seed  = 0);

int   STORMAPI SStrNCat(char *dest, const char *src, DWORD max_length);

/*  SStrLen @ 506
 *  
 *  Retrieves the length of a string.
 *
 *  string:   The input string of which to obtain a
 *            length for.
 *
 *  Returns the length of the string.
 */
int
STORMAPI
SStrLen(
      const char *string);

/*  SStrCmp @ 508
 *  
 *  Compares two strings case sensitive.
 *
 *  string1:  The first string.
 *  string2:  The second string.
 *  size:     The maximum amount of characters to compare.
 *
 *  Returns 0 if strings are equal. See strcmp documentation for more details.
 */
int
STORMAPI
SStrCmp(
      const char *string1, 
      const char *string2, 
      size_t size);

#define SSCmp(s,x) ( SStrCmp(s,x,__STORM_SSIZEMIN(s,x)) )

/*  SStrCmpI @ 509
 *  
 *  Compares two strings case insensitive.
 *
 *  string1:  The first string.
 *  string2:  The second string.
 *  size:     The maximum amount of characters to compare.
 *
 *  Returns 0 if strings are equal. See strcmpi documentation for more details.
 */
int
STORMAPI
SStrCmpI(
      const char *string1, 
      const char *string2, 
      size_t size);

#define SSCmpI(s,x) ( SStrCmpI(s,x,__STORM_SSIZEMIN(s,x)) )

/*  SStrUpper @ 510
 *  
 *  Converts all lower-case alpha characters of a string to upper-case.
 *
 *  string:   The string to convert.
 *  
 *  Returns the same pointer given in the input.
 */
char*
STORMAPI
SStrUpper(
      char* string);

void  STORMAPI SRgn523(HANDLE hRgn, RECT *pRect, int a3, int a4);
void  STORMAPI SRgnCreateRegion(HANDLE *hRgn, int a2);
void  STORMAPI SRgnDeleteRegion(HANDLE hRgn);

void  STORMAPI SRgn529i(int handle, int a2, int a3);


/* SErrDisplayErrorFmt @ 562
 *
 * Displays a formatted error message. The message is detailed and flexible for many applications.
 * The message will be different if there is a debugger attached. Will typically terminate the application
 * unless the option to continue is given.
 *
 *  dwErrMessage:   The error code. See SErrGetLastError and GetLastError.
 *  logfilename:    The name of the file or object that this call belongs to.
 *  logline:        The line in the file or one of the SLOG_ macros.
 *  allowOption:    If TRUE, allows the user the option to continue execution, otherwise the program will terminate.
 *  exitCode:       The exit code used for program termination.
 *  format:         Additional message formatting. See printf.
 *
 *  Returns TRUE if the user chose to continue execution, FALSE otherwise.
 */
BOOL
SErrDisplayErrorFmt(
    DWORD dwErrMsg,
    const char *logfilename,
    int logline,
    BOOL allowOption,
    int exitCode,
    const char *format,
    ...);

#define SEDisplayErrorFmt(err,...) SErrDisplayErrorFmt(err, __FILE__, __LINE__, FALSE, 1, __VA_ARGS__)

/*  SErrCatchUnhandledExceptions @ 567
 *  
 *  Registers a top-level exception filter managed entirely by Storm.
 *  The registered filter will display formatted exception information by calling SErrDisplayError.
 */
void
STORMAPI
SErrCatchUnhandledExceptions();


/*  SStrChr @ 571
 *  
 *  Searches a string for the given character. See 
 *  strchr documentation for more details.
 *
 *  string:   The string to search.
 *  c:        The character to search for.
 *  
 *  Returns a pointer to the first occurance of the character.
 */
char*
STORMAPI
SStrChr(
    const char *string, 
    char c);


char *STORMAPI SStrChrR(const char *string, char c);


/*  SStrVPrintf @ 578
 *  
 *  Prints a formatted string to a destination buffer.
 *  This function calls vsnprintf with some extra error handling.
 *  See online documentation of vsnprintf for more details.
 *
 *  dest:   The destination buffer.
 *  size:   The size of the destination buffer.
 *  format: The format to use.
 *
 *  Returns the number of characters written.
 */
size_t
SStrVPrintf(
    char *dest, 
    size_t size, 
    const char *format, ...);


int STORMAPI SBigDel(void *buffer);

int STORMAPI SBigFromBinary(void *buffer, const void *str, size_t size);

int STORMAPI SBigNew(void **buffer);

int STORMAPI SBigPowMod(void *buffer1, void *buffer2, int a3, int a4);

int STORMAPI SBigToBinaryBuffer(void *buffer, int length, int a3, int a4);

#ifdef __cplusplus
/////////////////////////////////////////////////////////////////////////
/// High Level wrappers
#include <string>

namespace Storm
{
  class CFile;
  class CArchive;

  class CFile
  {
  private:
    HANDLE hFile;
    bool valid;
  public:
    CFile()
      : hFile(nullptr)
      , valid(false)
    {}

    CFile(const std::string &sFileName, unsigned flags = SFILE_FROM_MPQ, HANDLE hMpq = nullptr)
      : hFile(nullptr)
      , valid( SFileOpenFileEx(hMpq, sFileName.c_str(), flags, &hFile) != FALSE )
    { }

    CFile(CFile &&other)
      : hFile( other.hFile )
      , valid( other.valid )
    {
      other.valid = false;
      other.hFile = nullptr;
    }

    ~CFile()
    {
      close();
    }

    HANDLE handle() const
    {
      return hFile;
    }

    bool open(const std::string &sFileName, unsigned flags = SFILE_FROM_MPQ, HANDLE hMpq = nullptr)
    {
      valid = SFileOpenFileEx(hMpq, sFileName.c_str(), flags, &hFile) != FALSE;
      return isValid();
    }

    bool close()
    {
      bool result = false;
      if (isValid())
        result = SFileCloseFile(hFile) != FALSE;
      hFile = nullptr;
      return result;
    }

    size_t size() const
    {
      return SFileGetFileSize(hFile, nullptr);
    }

    bool read(void *buffer, size_t buffSize) const
    {
      DWORD read = 0;
      BOOL result = SFileReadFile(hFile, buffer, buffSize, &read, 0);
      
      return result != FALSE && read == buffSize;
    }

    bool isValid() const { return valid && hFile; }
    explicit operator bool() const { return isValid(); }
    bool operator !() const { return !isValid(); }
  };


  class CArchive
  {
  private:
    HANDLE hMpq;
    bool valid;
  public:
    CArchive()
      : hMpq(nullptr)
      , valid(false)
    {}

    CArchive(const std::string &sArchiveFileName, unsigned flags = 0, unsigned priority = 0)
      : hMpq(nullptr)
      , valid( SFileOpenArchive(sArchiveFileName.c_str(), priority, flags, &hMpq) != FALSE )
    { }

    CArchive(CArchive &&other)
      : hMpq(other.hMpq)
      , valid(other.valid)
    {
      other.valid = false;
    }

    ~CArchive()
    {
      close();
    }

    bool open(const std::string &sArchiveFileName, unsigned flags = 0, unsigned priority = 0)
    {
      valid = SFileOpenArchive(sArchiveFileName.c_str(), priority, flags, &hMpq) != FALSE;
      return isValid();
    }

    bool close()
    {
      bool result = false;
      if (isValid())
        result = SFileCloseArchive(hMpq) != FALSE;
      hMpq = nullptr;
      return result;
    }

    HANDLE handle() const
    {
      return hMpq;
    }

    bool isValid() const { return valid && hMpq; }
    explicit operator bool() const { return isValid(); }
    bool operator !() const { return !isValid(); }

    CFile openFile( const std::string &sFileName ) const
    {
      return CFile(sFileName, SFILE_FROM_MPQ, hMpq);
    }
  };

};

#endif


#endif
